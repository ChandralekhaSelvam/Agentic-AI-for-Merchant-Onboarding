"""
================================================================================
MERCHANT ONBOARDING PIPELINE — UI Server
================================================================================
Wraps the existing pipeline with Server-Sent Events (SSE) streaming so the
frontend can show real-time agent progress.

Run:
    cd <your-project-root>
    python merchant-ui/server.py

Then open: http://localhost:5050
================================================================================
"""

import os
import sys
import json
import queue
import threading
import time
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Dict, Any, Optional

from flask import Flask, Response, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Point to your agents folder ──────────────────────────────────────────────
AGENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agents")
sys.path.insert(0, os.path.abspath(AGENTS_DIR))

from dotenv import load_dotenv
load_dotenv(os.path.join(AGENTS_DIR, ".env"))

# ── Flask setup ───────────────────────────────────────────────────────────────
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
app = Flask(__name__, static_folder=UI_DIR)
CORS(app)

# =============================================================================
# EVENTED PIPELINE WRAPPER
# =============================================================================

def _ts():
    return datetime.now(timezone.utc).isoformat()

def _safe_dict(obj):
    """Convert dataclasses / non-serializable objects to dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_dict(i) for i in obj]
    return obj


class EventedPipeline:
    """
    Wraps MerchantOnboardingPipeline and emits SSE events at each step.
    Avoids modifying the original agent files.
    """

    AGENT_NAMES = {
        1: "Data ingestion",
        2: "Validation",
        3: "Risk & compliance",
        4: "Document requirements",
        5: "Negative news detector",
        6: "Sentiment analyzer",
        7: "Orchestration",
        8: "Summarization",
    }

    def __init__(self, event_queue: queue.Queue):
        self.eq = event_queue
        self._import_pipeline()

    def _import_pipeline(self):
        """Lazy import so Flask can start even if agents have import-time side effects."""
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

        self._DataIngestionAgent = DataIngestionAgent
        self._ValidationAgent = ValidationAgent
        self._run_risk_assessment = run_risk_assessment
        self._run_document_requirements = run_document_requirements
        self._Agent5 = Agent5NegativeNewsDetector
        self._Agent6 = Agent6CustomerSentimentAnalyzer
        self._Agent7 = Agent7OrchestrationAgent
        self._Agent3Output = Agent3Output
        self._Agent4Output = Agent4Output
        self._Agent5Output = Agent5Output
        self._Agent6Output = Agent6Output
        self._generate_summary = generate_onboarding_summary

    # ── Event emitter ────────────────────────────────────────────────────────

    def emit(self, event_type: str, **kwargs):
        self.eq.put({"type": event_type, "timestamp": _ts(), **kwargs})

    def emit_start(self, agent_id: int):
        self.emit("agent_start", agent_id=agent_id, agent_name=self.AGENT_NAMES[agent_id])

    def emit_complete(self, agent_id: int, output: Any):
        self.emit(
            "agent_complete",
            agent_id=agent_id,
            agent_name=self.AGENT_NAMES[agent_id],
            output=_safe_dict(output),
        )

    def emit_error(self, agent_id: int, error: str):
        self.emit("agent_error", agent_id=agent_id, agent_name=self.AGENT_NAMES[agent_id], error=error)

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self, raw_profile: Dict[str, Any]) -> Dict[str, Any]:
        import re
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.emit("pipeline_start", merchant_name=raw_profile.get("merchant_name", "Unknown"))

        # ── Agent 1 ───────────────────────────────────────────────────────────
        self.emit_start(1)
        try:
            db_path = os.path.join(AGENTS_DIR, "data", "merchants.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            ingestion_agent = self._DataIngestionAgent(db_path=db_path)
            a1_out = ingestion_agent.insert_merchant(
                name=raw_profile.get("merchant_name", ""),
                address=raw_profile.get("address", ""),
                website=raw_profile.get("website", ""),
                industry=raw_profile.get("industry", ""),
                mcc=raw_profile.get("mcc", ""),
            )
            self.emit_complete(1, a1_out)
        except Exception as e:
            a1_out = raw_profile
            self.emit_error(1, str(e))

        # Build the profile dict that downstream agents expect
        profile = {
            "merchant_name": raw_profile.get("merchant_name", ""),
            "address": raw_profile.get("address", ""),
            "website": raw_profile.get("website", ""),
            "industry": raw_profile.get("industry", ""),
            "mcc": raw_profile.get("mcc", ""),
        }

        # ── Agent 2 — Validation ─────────────────────────────────────────────
        self.emit_start(2)
        try:
            validation_agent = self._ValidationAgent()
            a2_out = validation_agent.validate(profile)
            validated_profile = a2_out.get("validated_profile", profile)
            self.emit_complete(2, a2_out)

            if validated_profile.get("status") == "invalid":
                self.emit("pipeline_error",
                          error=f"Validation failed (score={validated_profile.get('data_quality_score', 0)}). Pipeline stopped.")
                return {"error": "Validation failed", "detail": a2_out}
        except Exception as e:
            validated_profile = profile
            validated_profile["status"] = "partial"
            self.emit_error(2, str(e))
            a2_out = {"validated_profile": validated_profile, "error": str(e)}

        # ── Agents 3, 5, 6 — Parallel ────────────────────────────────────────
        api_key = os.getenv("OPENAI_API_KEY", "")
        merchant_name = validated_profile.get("merchant_name", profile["merchant_name"])
        url = validated_profile.get("website", profile["website"])
        industry = validated_profile.get("industry", profile["industry"])

        a3_out = a5_out = a6_out = {}

        parallel_done = threading.Event()

        def run_agent3():
            self.emit_start(3)
            try:
                out = self._run_risk_assessment(validated_profile)
                self.emit_complete(3, out)
                return out
            except Exception as e:
                self.emit_error(3, str(e))
                return {"risk_level": "MEDIUM", "error": str(e)}

        def run_agent5():
            self.emit_start(5)
            try:
                agent = self._Agent5(
                    merchant_name=merchant_name,
                    merchant_url=url,
                    industry=industry,
                    openai_api_key=api_key,
                )
                out = agent.process()
                out_dict = _safe_dict(out)
                self.emit_complete(5, out_dict)
                return out_dict
            except Exception as e:
                self.emit_error(5, str(e))
                return {"overall_risk_severity": "NONE", "overall_risk_score": 0.0,
                        "recommended_action": "MANUAL_REVIEW", "error": str(e)}

        def run_agent6():
            self.emit_start(6)
            try:
                agent = self._Agent6(
                    merchant_name=merchant_name,
                    merchant_url=url,
                    industry=industry,
                )
                out = agent.process()
                out_dict = _safe_dict(out)
                self.emit_complete(6, out_dict)
                return out_dict
            except Exception as e:
                self.emit_error(6, str(e))
                return {"overall_sentiment_score": 0.0, "customer_satisfaction_rating": 0.0,
                        "recommendation_flag": "REVIEW", "error": str(e)}

        with ThreadPoolExecutor(max_workers=3) as ex:
            future_map = {
                ex.submit(run_agent3): "a3",
                ex.submit(run_agent5): "a5",
                ex.submit(run_agent6): "a6",
            }
            results = {}
            for f in as_completed(future_map):
                key = future_map[f]
                results[key] = f.result()

        a3_out = results.get("a3", {})
        a5_out = results.get("a5", {})
        a6_out = results.get("a6", {})

        # ── Agent 4 — Documents (after Agent 3) ──────────────────────────────
        self.emit_start(4)
        try:
            risk_level = str(a3_out.get("risk_level", "MEDIUM")).upper()
            state = {
                "merchant_name": merchant_name,
                "merchant_url": url,
                "industry": industry,
                "risk_review": "yes" if risk_level in ("MEDIUM", "HIGH") else "no",
                "compliance_review": "yes" if (
                    a3_out.get("compliance_review_required", False) or risk_level == "HIGH"
                ) else "no",
            }
            a4_out = self._run_document_requirements(state)
            self.emit_complete(4, a4_out)
        except Exception as e:
            a4_out = {"risk_output": {}, "compliance_output": {"compliance_documentation": {}}, "error": str(e)}
            self.emit_error(4, str(e))

        # ── Agent 7 — Orchestration ───────────────────────────────────────────
        self.emit_start(7)
        try:
            a7_agent = self._Agent7()

            # Parse agent 3
            def parse_a3(raw):
                if raw.get("risk_level"):
                    return raw
                text = raw.get("raw_output", "")
                parsed = {"risk_level": "MEDIUM"}
                m = re.search(r"\*?\*?Risk Level\*?\*?\s*[:\-]?\s*\*?\*?(Low|Medium|High)", text, re.IGNORECASE)
                if m: parsed["risk_level"] = m.group(1).upper()
                parsed["high_risk_flag"] = bool(re.search(r"High Risk Flag.*?Yes", text, re.IGNORECASE))
                parsed["compliance_review_required"] = bool(re.search(r"Compliance Review Required.*?Yes", text, re.IGNORECASE))
                parsed["tpb_threshold_exceeded"] = bool(re.search(r"TPB Review Required.*?Yes", text, re.IGNORECASE))
                parsed["required_questionnaires"] = sorted(set(re.findall(r"Q-\d{3}", text)))
                parsed["applicable_playbooks"] = sorted(set(re.findall(r"PB-\d{3}", text)))
                m2 = re.search(r"\*?\*?Industry\*?\*?\s*[:\-]?\s*\*?\*?([A-Za-z][^\n*]+)", text)
                parsed["industry_risk_category"] = m2.group(1).strip() if m2 else ""
                return parsed

            a3p = parse_a3(a3_out)
            risk_out = a4_out.get("risk_output", {}) or {}
            comp_out = a4_out.get("compliance_output", {}) or {}
            comp_doc = comp_out.get("compliance_documentation", {}) if isinstance(comp_out, dict) else {}

            agent3 = self._Agent3Output(
                merchant_name=merchant_name,
                risk_level=a3p.get("risk_level", "MEDIUM"),
                high_risk_flag=a3p.get("high_risk_flag", False),
                industry_risk_category=a3p.get("industry_risk_category", ""),
                required_questionnaires=a3p.get("required_questionnaires", []),
                applicable_playbooks=a3p.get("applicable_playbooks", []),
                compliance_review_required=a3p.get("compliance_review_required", False),
                tpb_threshold_exceeded=a3p.get("tpb_threshold_exceeded", False),
            )

            agent4 = self._Agent4Output(
                merchant_name=a4_out.get("merchant_name", merchant_name),
                merchant_type=risk_out.get("merchant_type", "Unknown"),
                risk_review=risk_out.get("risk_review", "Unknown"),
                pd_score=risk_out.get("pd_score"),
                documents_required=risk_out.get("documents_required", []),
                merchant_industry=comp_doc.get("merchant_industry", "") if isinstance(comp_doc, dict) else "",
                high_risk_match=bool(comp_doc.get("high_risk_match", False)) if isinstance(comp_doc, dict) else False,
                artifacts_required=comp_doc.get("artifacts_required", []) if isinstance(comp_doc, dict) else [],
                compliance_message=comp_doc.get("message", "") if isinstance(comp_doc, dict) else "",
            )

            a5_score = float(a5_out.get("overall_risk_score", 0.0))
            a5_sev = ("CRITICAL" if a5_score >= 76 else "HIGH" if a5_score >= 51
                      else "MEDIUM" if a5_score >= 26 else "LOW" if a5_score > 0 else "NONE")

            agent5 = self._Agent5Output(
                merchant_name=merchant_name,
                overall_risk_severity=a5_sev,
                overall_risk_score=a5_score,
                recommended_action=a5_out.get("recommended_action", "APPROVE"),
                data_sufficiency=a5_out.get("data_sufficiency", "SUFFICIENT"),
                findings_count=len(a5_out.get("findings", [])),
                corroboration_score=float(a5_out.get("corroboration_score", 0.0)),
            )

            agent6 = self._Agent6Output(
                merchant_name=merchant_name,
                overall_sentiment_score=float(a6_out.get("overall_sentiment_score", 0.0)),
                customer_satisfaction_rating=float(a6_out.get("customer_satisfaction_rating", 0.0)),
                recommendation_flag=a6_out.get("recommendation_flag", "APPROVE"),
                review_count=int(a6_out.get("review_count", 0)),
                negative_ratio=float(a6_out.get("sentiment_distribution", {}).get("negative", 0.0)
                                     if isinstance(a6_out.get("sentiment_distribution"), dict)
                                     else a6_out.get("negative_ratio", 0.0)),
            )

            a7_out = a7_agent.process(
                merchant_name=merchant_name,
                agent3_output=agent3,
                agent4_output=agent4,
                agent5_output=agent5,
                agent6_output=agent6,
            )
            self.emit_complete(7, _safe_dict(a7_out))
        except Exception as e:
            a7_out = {"final_decision": "MANUAL_REVIEW", "decision_rationale": str(e), "error": str(e)}
            self.emit_error(7, str(e))

        # ── Agent 8 — Summarization ───────────────────────────────────────────
        self.emit_start(8)
        try:
            a8_out = self._generate_summary(
                orchestration_decision=_safe_dict(a7_out),
                evidence_metadata={
                    "risk_assessment": a3_out,
                    "document_requirements": a4_out,
                    "adverse_media": a5_out,
                    "sentiment": a6_out,
                },
            )
            self.emit_complete(8, a8_out if isinstance(a8_out, dict) else {"summary": str(a8_out)})
        except Exception as e:
            a8_out = {"error": str(e)}
            self.emit_error(8, str(e))

        # ── Final report ──────────────────────────────────────────────────────
        final_report = {
            "merchant_name": merchant_name,
            "final_decision": _safe_dict(a7_out).get("final_decision", "MANUAL_REVIEW"),
            "decision_rationale": _safe_dict(a7_out).get("decision_rationale", ""),
            "escalation_reasons": _safe_dict(a7_out).get("escalation_reasons", []),
            "pipeline_timestamp": _ts(),
            "agent_outputs": {
                "agent_1": _safe_dict(a1_out),
                "agent_2": _safe_dict(a2_out),
                "agent_3": _safe_dict(a3_out),
                "agent_4": _safe_dict(a4_out),
                "agent_5": _safe_dict(a5_out),
                "agent_6": _safe_dict(a6_out),
                "agent_7": _safe_dict(a7_out),
                "agent_8": _safe_dict(a8_out),
            },
        }

        self.emit("pipeline_complete", report=final_report)
        return final_report


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")

@app.route("/api/run", methods=["POST"])
def run_pipeline():
    data = request.get_json(force=True)
    if not data or not data.get("merchant_name"):
        return jsonify({"error": "merchant_name is required"}), 400

    eq = queue.Queue()

    def worker():
        try:
            pipeline = EventedPipeline(eq)
            pipeline.run(data)
        except Exception as e:
            eq.put({"type": "pipeline_error", "timestamp": _ts(), "error": str(e)})
        finally:
            eq.put(None)  # sentinel

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    def stream():
        while True:
            try:
                msg = eq.get(timeout=180)
            except queue.Empty:
                yield "data: {\"type\":\"heartbeat\"}\n\n"
                continue
            if msg is None:
                break
            try:
                yield f"data: {json.dumps(msg, default=str)}\n\n"
            except Exception:
                pass

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": _ts()})

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    print(f"\n{'='*60}")
    print(f"  Merchant Onboarding UI Server")
    print(f"  http://localhost:{port}")
    print(f"  Agents dir: {AGENTS_DIR}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
