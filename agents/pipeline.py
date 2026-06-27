"""
================================================================================
MERCHANT ONBOARDING PIPELINE — End-to-End Orchestrator
================================================================================
Runs the full agent pipeline for a single merchant:

    Agent 1 (input) → Agent 2 (validation)
        → [PARALLEL] Agent 3 + Agent 5 + Agent 6
        → Agent 4 (depends on Agent 3 output)
        → Agent 7 (orchestration)
        → Agent 8 (summarization)
        → Final onboarding report

This is the ONLY file you should run in production:
    python pipeline.py
================================================================================
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# ─── Import all agents ──────────────────────────────────────────────
from agent_1_data_ingestion import DataIngestionAgent
from agent_2_validation_agent import ValidationAgent
from agent_3_risk_compliance import run_risk_assessment
from agent_4_document_req_decision_logic_agent import run_document_requirements
from agent_5_negative_news_detector import Agent5NegativeNewsDetector
from agent_6_sentiment_analyzer import Agent6CustomerSentimentAnalyzer
from agent_7_orchestration_agent import (
    Agent7OrchestrationAgent,
    Agent3Output, Agent4Output, Agent5Output, Agent6Output,
)
from agent_8_summarization_agent import generate_onboarding_summary

# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class MerchantOnboardingPipeline:
    """
    End-to-end merchant onboarding pipeline.

    Execution order:
        1. Agent 2 (Validation) — sequential
        2. Agents 3, 5, 6 — PARALLEL (independent of each other)
        3. Agent 4 — sequential after Agent 3 (depends on risk_level)
        4. Agent 7 (Orchestration) — sequential
        5. Agent 8 (Summarization) — sequential
    """

    def __init__(self, openai_api_key: str = ""):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable")

        self.validation_agent = ValidationAgent()
        self.orchestration_agent = Agent7OrchestrationAgent()
        self.trace: list = []

    def _log(self, msg: str):
        entry = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        self.trace.append(entry)
        print(entry)

    def run(self, raw_merchant_profile: Dict[str, Any]) -> Dict[str, Any]:
        pipeline_start = time.time()
        self.trace = []

        print(f"\n{'='*70}")
        print(f"  MERCHANT ONBOARDING PIPELINE")
        print(f"  Merchant: {raw_merchant_profile.get('merchant_name', 'Unknown')}")
        print(f"{'='*70}\n")

        # ─── STEP 1: Agent 2 — Validation ─────────────────────────────
        self._log("STEP 1: Running Agent 2 (Validation)...")
        validation_result = self.validation_agent.validate(raw_merchant_profile)
        validated_profile = validation_result["validated_profile"]

        if validated_profile.get("status") == "invalid":
            self._log(
                f"⚠ Validation failed (score={validated_profile['data_quality_score']}). "
                f"Returning early."
            )
            return self._build_rejection(
                raw_merchant_profile,
                reason="Validation failed - Critical data missing",
                validation_result=validation_result,
            )

        if validated_profile.get("status") == "partial":
            self._log(f"⚠ Validation passed with warnings (score={validated_profile['data_quality_score']})")
        else:
            self._log(f"✓ Validation passed (score={validated_profile['data_quality_score']})")

        # ─── STEP 2: Run Agents 3, 5, 6 in parallel, then Agent 4 ─────
        self._log("STEP 2: Running Agents 3, 5, 6 in parallel, then Agent 4...")
        agent_outputs = self._run_parallel_agents(validated_profile)

        # ─── STEP 3: Agent 7 — Orchestration ──────────────────────────
        self._log("STEP 3: Running Agent 7 (Orchestration)...")
        orchestration_result = self._run_orchestration(
            merchant_name=validated_profile["merchant_name"],
            agent_outputs=agent_outputs,
        )
        self._log(f"✓ Orchestration decision: {orchestration_result['final_decision']}")

        # ─── STEP 4: Agent 8 — Summarization ──────────────────────────
        self._log("STEP 4: Running Agent 8 (Summarization)...")
        summary = generate_onboarding_summary(
            orchestration_decision=orchestration_result,
            evidence_metadata={
                "risk_assessment": agent_outputs["agent_3"],
                "document_requirements": agent_outputs["agent_4"],
                "adverse_media": agent_outputs["agent_5"],
                "sentiment": agent_outputs["agent_6"],
            },
        )
        self._log("✓ Summary generated")

        # ─── STEP 5: Build Final Report ───────────────────────────────
        elapsed = round(time.time() - pipeline_start, 2)
        self._log(f"Pipeline complete in {elapsed}s")

        final_report = {
            "merchant_name": validated_profile["merchant_name"],
            "pipeline_timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_duration_seconds": elapsed,
            "final_decision": orchestration_result["final_decision"],
            "decision_rationale": orchestration_result["decision_rationale"],
            "escalation_reasons": orchestration_result.get("escalation_reasons", []),
            "required_actions": orchestration_result.get("required_actions", []),
            "agent_outputs": {
                "agent_2_validation": validation_result,
                "agent_3_risk": agent_outputs["agent_3"],
                "agent_4_documents": agent_outputs["agent_4"],
                "agent_5_adverse_media": agent_outputs["agent_5"],
                "agent_6_sentiment": agent_outputs["agent_6"],
                "agent_7_orchestration": orchestration_result,
                "agent_8_summary": summary,
            },
            "pipeline_trace": self.trace,
        }

        self._print_final_report(final_report)
        return final_report

    # ─────────────────────────────────────────────────────────────────
    # PARALLEL + SEQUENTIAL EXECUTION
    # ─────────────────────────────────────────────────────────────────

    def _run_parallel_agents(self, validated_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1: Run Agents 3, 5, 6 in parallel (they're independent).
        Stage 2: Run Agent 4 sequentially after Agent 3.
        """
        merchant_name = validated_profile["merchant_name"]
        industry = validated_profile.get("industry", "")
        url = validated_profile.get("website", "")

        results: Dict[str, Any] = {}

        # ── Stage 1: Agents 3, 5, 6 in parallel ───────────────────────
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_agent = {
                executor.submit(self._safe_run_agent_3, validated_profile): "agent_3",
                executor.submit(self._safe_run_agent_5, merchant_name, url, industry): "agent_5",
                executor.submit(self._safe_run_agent_6, merchant_name, url, industry): "agent_6",
            }

            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                    self._log(f"  ✓ {agent_name} completed")
                except Exception as e:
                    self._log(f"  ✗ {agent_name} failed: {e}")
                    results[agent_name] = self._empty_agent_output(agent_name, merchant_name)

        # ── Stage 2: Agent 4 after Agent 3 ────────────────────────────
        self._log("  Running Agent 4 (depends on Agent 3 output)...")
        try:
            results["agent_4"] = self._safe_run_agent_4(
                validated_profile, results.get("agent_3")
            )
            self._log("  ✓ agent_4 completed")
        except Exception as e:
            self._log(f"  ✗ agent_4 failed: {e}")
            results["agent_4"] = self._empty_agent_output("agent_4", merchant_name)

        return results

    # ─────────────────────────────────────────────────────────────────
    # SAFE AGENT WRAPPERS
    # ─────────────────────────────────────────────────────────────────

    def _safe_run_agent_3(self, validated_profile: Dict[str, Any]) -> Dict[str, Any]:
        return run_risk_assessment(validated_profile)

    def _safe_run_agent_4(self, validated_profile: Dict[str, Any],
                           risk_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Wrapper: Agent 4 — Document Requirements & Decision Logic.

        Returns Agent 4's raw nested output (risk_output + compliance_output).
        The flattening into Agent4Output happens in _run_orchestration.
        """
        if risk_result is None:
            risk_result = {"risk_level": "MEDIUM"}

        risk_level = str(risk_result.get("risk_level", "MEDIUM")).upper()

        risk_review_flag = "yes" if risk_level in ("MEDIUM", "HIGH") else "no"
        compliance_review_flag = (
            "yes" if (
                risk_result.get("compliance_review_required", False)
                or risk_level == "HIGH"
            )
            else "no"
        )

        state = {
            "merchant_name": validated_profile.get("merchant_name", ""),
            "merchant_url": validated_profile.get("website", ""),
            "industry": validated_profile.get("industry", ""),
            "risk_review": risk_review_flag,
            "compliance_review": compliance_review_flag,
        }

        return run_document_requirements(state)

    def _safe_run_agent_5(self, name: str, url: str, industry: str) -> Dict[str, Any]:
        agent = Agent5NegativeNewsDetector(
            merchant_name=name,
            merchant_url=url,
            industry=industry,
            openai_api_key=self.openai_api_key,
        )
        output = agent.process()
        return asdict(output) if hasattr(output, '__dataclass_fields__') else output

    def _safe_run_agent_6(self, name: str, url: str, industry: str) -> Dict[str, Any]:
        agent = Agent6CustomerSentimentAnalyzer(
            merchant_name=name,
            merchant_url=url,
            industry=industry,
        )
        output = agent.process()
        return asdict(output) if hasattr(output, '__dataclass_fields__') else output

    def _empty_agent_output(self, agent_name: str, merchant_name: str) -> Dict[str, Any]:
        return {
            "merchant_name": merchant_name,
            "error": f"{agent_name} failed to produce output",
            "overall_risk_severity": "NONE",
            "overall_risk_score": 0.0,
            "recommended_action": "MANUAL_REVIEW",
            "overall_sentiment_score": 0.0,
            "customer_satisfaction_rating": 0.0,
            "recommendation_flag": "REVIEW",
            "risk_level": "MEDIUM",
            "risk_output": {},
            "compliance_output": {"compliance_documentation": {}},
        }

    # ─────────────────────────────────────────────────────────────────
    # AGENT 3 RAW OUTPUT PARSER
    # ─────────────────────────────────────────────────────────────────

    def _parse_agent3_output(self, a3_raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent 3 sometimes returns {"raw_output": "markdown text"}.
        This extracts the structured fields Agent 7 needs.
        """
        if "risk_level" in a3_raw and a3_raw.get("risk_level"):
            return a3_raw

        raw_text = a3_raw.get("raw_output", "")
        if not raw_text:
            self._log("  ⚠ Agent 3 returned no output — defaulting to MEDIUM")
            return {"risk_level": "MEDIUM"}

        parsed: Dict[str, Any] = {"raw_output": raw_text}

        m = re.search(
            r"\*?\*?Risk Level\*?\*?\s*[:\-]?\s*\*?\*?(Low|Medium|High)",
            raw_text, re.IGNORECASE,
        )
        parsed["risk_level"] = m.group(1).upper() if m else "MEDIUM"

        m = re.search(
            r"\*?\*?Risk Score\*?\*?\s*[:\-]?\s*\*?\*?(\d+)",
            raw_text, re.IGNORECASE,
        )
        parsed["risk_score"] = int(m.group(1)) if m else 50

        m = re.search(
            r"\*?\*?High Risk Flag\*?\*?\s*[:\-]?\s*\*?\*?(Yes|No)",
            raw_text, re.IGNORECASE,
        )
        parsed["high_risk_flag"] = (m is not None and m.group(1).lower() == "yes")

        m = re.search(
            r"\*?\*?Compliance Review Required\*?\*?\s*[:\-]?\s*\*?\*?(Yes|No)",
            raw_text, re.IGNORECASE,
        )
        parsed["compliance_review_required"] = (
            m is not None and m.group(1).lower() == "yes"
        )

        m = re.search(
            r"\*?\*?TPB Review Required\*?\*?\s*[:\-]?\s*\*?\*?(Yes|No)",
            raw_text, re.IGNORECASE,
        )
        parsed["tpb_threshold_exceeded"] = (
            m is not None and m.group(1).lower() == "yes"
        )

        parsed["required_questionnaires"] = sorted(set(re.findall(r"Q-\d{3}", raw_text)))
        parsed["applicable_playbooks"] = sorted(set(re.findall(r"PB-\d{3}", raw_text)))

        m = re.search(
            r"\*?\*?Industry\*?\*?\s*[:\-]?\s*\*?\*?([A-Za-z][^\n*]+)",
            raw_text,
        )
        parsed["industry_risk_category"] = m.group(1).strip() if m else ""

        self._log(
            f"  Parsed Agent 3: risk={parsed['risk_level']} "
            f"score={parsed['risk_score']} "
            f"Q={len(parsed['required_questionnaires'])} "
            f"PB={len(parsed['applicable_playbooks'])}"
        )

        return parsed

    # ─────────────────────────────────────────────────────────────────
    # ORCHESTRATION INVOCATION
    # ─────────────────────────────────────────────────────────────────

    def _run_orchestration(self, merchant_name: str,
                            agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Agent 3/4/5/6 outputs into the dataclasses Agent 7 expects,
        then run the orchestration graph.
        """
        a3 = self._parse_agent3_output(agent_outputs["agent_3"])
        a4_raw = agent_outputs["agent_4"]
        a5 = agent_outputs["agent_5"]
        a6 = agent_outputs["agent_6"]

        # ── Unwrap Agent 4's nested structure ────────────────────────
        risk_output = a4_raw.get("risk_output", {}) or {}
        compliance_output = a4_raw.get("compliance_output", {}) or {}
        compliance_doc = compliance_output.get("compliance_documentation", {})

        if isinstance(compliance_doc, dict):
            merchant_industry = compliance_doc.get("merchant_industry", "")
            high_risk_match = bool(compliance_doc.get("high_risk_match", False))
            artifacts = compliance_doc.get("artifacts_required", [])
            if not isinstance(artifacts, list):
                artifacts = []
            compliance_message = compliance_doc.get("message", "")
        else:
            merchant_industry = ""
            high_risk_match = False
            artifacts = []
            compliance_message = str(compliance_doc) if compliance_doc else ""

        # ── Build Agent 3 dataclass ──────────────────────────────────
        agent3_output = Agent3Output(
            merchant_name=merchant_name,
            risk_level=a3.get("risk_level", "MEDIUM"),
            high_risk_flag=a3.get("high_risk_flag", False), 
            industry_risk_category=a3.get("industry_risk_category", ""),
            required_questionnaires=a3.get("required_questionnaires", []),
            applicable_playbooks=a3.get("applicable_playbooks", []),
            compliance_review_required=a3.get("compliance_review_required", False),
            tpb_threshold_exceeded=a3.get("tpb_threshold_exceeded", False),
        )

        # ── Build Agent 4 dataclass ──────────────────────────────────
        agent4_output = Agent4Output(
            merchant_name=a4_raw.get("merchant_name", merchant_name),
            merchant_type=risk_output.get("merchant_type", "Unknown"),
            risk_review=risk_output.get("risk_review", "Unknown"),
            pd_score=risk_output.get("pd_score"),
            documents_required=risk_output.get("documents_required", []),
            merchant_industry=merchant_industry,
            high_risk_match=high_risk_match,
            artifacts_required=artifacts,
            compliance_message=compliance_message,
        )

        # ── Derive Agent 5 severity from score (fixes label/score mismatch) ──
        a5_score = a5.get("overall_risk_score", 0.0)
        if a5_score >= 76:
            a5_severity = "CRITICAL"
        elif a5_score >= 51:
            a5_severity = "HIGH"
        elif a5_score >= 26:
            a5_severity = "MEDIUM"
        elif a5_score > 0:
            a5_severity = "LOW"
        else:
            a5_severity = "NONE"

        agent5_output = Agent5Output(
            merchant_name=merchant_name,
            overall_risk_severity=a5_severity,
            overall_risk_score=a5_score,
            recommended_action=a5.get("recommended_action", "APPROVE"),
            data_sufficiency=a5.get("data_sufficiency", "SUFFICIENT"),
            findings_count=len(a5.get("findings", [])),
            corroboration_score=a5.get("corroboration_score", 0.0),
        )

        # ── Build Agent 6 dataclass ──────────────────────────────────
        agent6_output = Agent6Output(
            merchant_name=merchant_name,
            overall_sentiment_score=a6.get("overall_sentiment_score", 0.0),
            customer_satisfaction_rating=a6.get("customer_satisfaction_rating", 0.0),
            recommendation_flag=a6.get("recommendation_flag", "APPROVE"),
            review_count=a6.get("review_count", 0),
            negative_ratio=a6.get("sentiment_distribution", {}).get("negative", 0.0),
        )

        return self.orchestration_agent.process(
            merchant_name=merchant_name,
            agent3_output=agent3_output,
            agent4_output=agent4_output,
            agent5_output=agent5_output,
            agent6_output=agent6_output,
        )

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _build_rejection(self, raw_profile: Dict[str, Any], reason: str,
                          **kwargs) -> Dict[str, Any]:
        return {
            "merchant_name": raw_profile.get("merchant_name", "Unknown"),
            "final_decision": "REJECT",
            "decision_rationale": reason,
            "pipeline_timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_trace": self.trace,
            **kwargs,
        }

    def _print_final_report(self, report: Dict[str, Any]):
        icon = {"APPROVE": "✅", "MANUAL_REVIEW": "⚠️", "REJECT": "🚫"}.get(
            report["final_decision"], "?"
        )
        print(f"\n{'='*70}")
        print(f"  FINAL ONBOARDING DECISION: {report['merchant_name']}")
        print(f"{'='*70}")
        print(f"  Decision:     {icon} {report['final_decision']}")
        print(f"  Duration:     {report.get('pipeline_duration_seconds', 'N/A')}s")
        print(f"\n  Rationale:")
        print(f"    {report.get('decision_rationale', 'N/A')}")
        if report.get("escalation_reasons"):
            print(f"\n  Escalation reasons:")
            for r in report["escalation_reasons"]:
                print(f"    • {r}")
        print(f"{'='*70}\n")

# Main entry point
if __name__ == "__main__":
    import os
    import json
    from agent_1_data_ingestion import DataIngestionAgent
    agent1 = DataIngestionAgent(db_path="data/merchants.db")
    
    # sample_merchant = agent1.insert_merchant(name="WhiteHat Jr", address="123 Main St", website="https://www.whitechatjr.com", industry="EdTech", mcc="8299")
    # sample_merchant = agent1.insert_merchant(name="PayPal", address="123 Main St", website="https://www.paypal.com", industry="Fintech", mcc="6012")
    sample_merchant = agent1.insert_merchant(name="BetKing Online Ltd.", address="123 Main St", website="https://m.betking.com/", industry="Online Gambling / Betting", mcc="7995")

    input_to_pipeline = {
        "merchant_name": sample_merchant["merchant_name"],
        "address": sample_merchant["address"],
        "website": sample_merchant["website"],
        "industry": sample_merchant["industry"],
        "mcc": sample_merchant["mcc"],
        "additional_details": sample_merchant.get("additional_data", "")
    }
    print(f"🚀 Starting pipeline for: {input_to_pipeline}")
    pipeline = MerchantOnboardingPipeline()
    result = pipeline.run(input_to_pipeline)

    merchant_filename = result["merchant_name"].replace(" ", "_").lower()
    filename = f"onboarding_{merchant_filename}_report.json"

    with open(filename, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"📁 Full report saved: {filename}")
   