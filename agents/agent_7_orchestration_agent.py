"""
================================================================================
AGENT 7 — ORCHESTRATION AGENT
================================================================================
Process (LangGraph state machine):
    1. INTAKE       — normalize and validate inputs from all upstream agents
    2. ASSESS       — evaluate each agent signal independently
    3. CONFLICT     — LLM detects disagreements between Agents 3, 5, 6
    4. DECIDE       — LLM applies policy rules to produce final decision
    5. FINALIZE     — produce final onboarding decision + audit trail
 
Both conflict detection and decision making are LLM-powered (gpt-4o-mini)
with rule-based fallbacks if the LLM is unavailable or returns malformed JSON.
================================================================================
"""

from dotenv import load_dotenv
load_dotenv()
import os
import re
import json
import hashlib
from typing import Dict, List, Optional, Any, TypedDict, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import StrEnum
 
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

_llm_client: Optional[ChatOpenAI] = None

def _get_llm() -> Optional[ChatOpenAI]:
    """Lazily initialize the LLM client. Returns None if no API key is set."""
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        _llm_client = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
            timeout=30,
        )
    return _llm_client

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from LLM output"""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None

# Enums - Policy aligned decisions
class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class Severity(StrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class FinalDecision(StrEnum):
    APPROVE = "APPROVE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECT = "REJECT"

class SentimentFlag(StrEnum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"

# Input Schemas
@dataclass
class Agent3Output:
    """Risk & Compliance Assessment Agent output."""
    merchant_name: str
    risk_level: str
    high_risk_flag: bool = False
    tpb_threshold_exceeded: bool = False
    compliance_review_required: bool = False
    industry_risk_category: str = ""
    required_questionnaires: List[str] = field(default_factory=list)
    applicable_playbooks: List[str] = field(default_factory=list)
    
@dataclass
class Agent4Output:
    """Document Requirements & Decision Logic Agent output."""
    merchant_name: str
    pd_score: Optional[float] = None
    documents_required: List[str] = field(default_factory=list)
    merchant_type: str = "Unknown"
    risk_review: str = "Unknown"
    merchant_industry: str = ""
    high_risk_match: bool = False
    artifacts_required: List[str] = field(default_factory=list)
    compliance_message: str = ""

    @property
    def is_existing_merchant(self) -> bool:
        return self.merchant_type == "Existing Merchant"
 
    @property
    def is_new_merchant(self) -> bool:
        return self.merchant_type == "New Merchant"

@dataclass
class Agent5Output:
    """Negative News Detection Agent output."""
    merchant_name: str
    overall_risk_severity: str               # NONE | LOW | MEDIUM | HIGH | CRITICAL
    overall_risk_score: float                # 0-100
    recommended_action: str                  # APPROVE | EDD | MANUAL_REVIEW | DECLINE
    data_sufficiency: str                    # SUFFICIENT | LIMITED | INSUFFICIENT
    findings_count: int = 0
    corroboration_score: float = 0.0         # 0-1

@dataclass
class Agent6Output:
    """Customer Sentiment & Review Analysis Agent output."""
    merchant_name: str
    overall_sentiment_score: float           # -1.0 to 1.0
    customer_satisfaction_rating: float      # 0-5
    recommendation_flag: str                 # APPROVE | REVIEW | REJECT
    review_count: int = 0
    negative_ratio: float = 0.0

# ORCHESTRATION STATE
class OrchestrationState(TypedDict):
    """State object that flows through the LangGraph nodes.

    Each node reads from and writes to this state. LangGraph merges updates
    automatically between node transitions.
    """
    # Inputs
    merchant_name: str
    agent3: Dict[str, Any]
    agent4: Dict[str, Any]
    agent5: Dict[str, Any]
    agent6: Dict[str, Any]

    # Intermediate signals (populated by assess node)
    signals: Dict[str, Any]

    # Conflict analysis
    conflicts: List[str]
    has_critical_conflict: bool

    # Final decision outputs
    final_decision: str
    escalation_reasons: List[str]
    decision_rationale: str
    required_actions: List[str]

    # Audit trail
    orchestration_id: str
    timestamp: str
    llm: Any
    processing_trace: List[str]

# Decision policy
class OrchestrationPolicy:
    """
    Encapsulates all policy-driven decision rules.
    """
    # Policy thresholds
    REJECT_SENTIMENT_THRESHOLD = -0.6
    REJECT_PD_SCORE_THRESHOLD = 0.7
    HIGH_RISK_ADVERSE_SCORE = 51
    CRITICAL_ADVERSE_SCORE = 76
    MEDIUM_RISK_ADVERSE_SCORE = 26

    @staticmethod
    def assess_agent3(agent3: Dict[str, Any]) -> Dict[str, Any]:
        """Extract normalized risk signals from Agent 3 output."""
        return {
            "risk_level": agent3.get("risk_level", "MEDIUM").upper(),
            "high_risk_flag": agent3.get("high_risk_flag", "False"),
            "compliance_required": agent3.get("compliance_review_required", False),
            "tpb_threshold_exceeded": agent3.get("tpb_threshold_exceeded", False),
        }

    @staticmethod
    def assess_agent4(agent4: Dict[str, Any]) -> Dict[str, Any]:
        """Extract document/PD/compliance signals from Agent 4 output."""
        pd = agent4.get("pd_score")
        return {
            "merchant_type": agent4.get("merchant_type", "Unknown"),
            "is_existing_merchant": agent4.get("merchant_type") == "Existing Merchant",
            "is_new_merchant": agent4.get("merchant_type") == "New Merchant",
            "risk_review": agent4.get("risk_review", "Unknown"),
            "pd_score": pd,
            "pd_exceeds_threshold": (
                pd is not None and pd >= OrchestrationPolicy.REJECT_PD_SCORE_THRESHOLD
            ),
            "document_count": len(agent4.get("documents_required", [])),
            "high_risk_match": agent4.get("high_risk_match", False),
            "compliance_artifacts_count": len(agent4.get("artifacts_required", [])),
            "compliance_message": agent4.get("compliance_message", ""),
            "merchant_industry": agent4.get("merchant_industry", ""),
        }

    @staticmethod
    def assess_agent5(agent5: Dict[str, Any]) -> Dict[str, Any]:
        """Extract adverse media signals from Agent 5 output."""
        severity = agent5.get("overall_risk_severity", "NONE").upper()
        score = agent5.get("overall_risk_score", 0.0)
        return {
            "severity": severity,
            "risk_score": score,
            "is_critical": severity == "CRITICAL" or score >= OrchestrationPolicy.CRITICAL_ADVERSE_SCORE,
            "is_high": severity == "HIGH" or score >= OrchestrationPolicy.HIGH_RISK_ADVERSE_SCORE,
            "is_medium": severity == "MEDIUM" or score >= OrchestrationPolicy.MEDIUM_RISK_ADVERSE_SCORE,
            "data_insufficient": agent5.get("data_sufficiency") == "INSUFFICIENT",
            "action": agent5.get("recommended_action", "APPROVE"),
        }

    @staticmethod
    def assess_agent6(agent6: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sentiment signals from Agent 6 output."""
        sentiment_score = agent6.get("overall_sentiment_score", 0.0)
        return {
            "sentiment_score": sentiment_score,
            "satisfaction": agent6.get("customer_satisfaction_rating", 0.0),
            "flag": agent6.get("recommendation_flag", "APPROVE").upper(),
            "is_very_negative": sentiment_score <= OrchestrationPolicy.REJECT_SENTIMENT_THRESHOLD,
            "review_count": agent6.get("review_count", 0),
        }

# LLM with rule based fallback
    @staticmethod
    def _detect_conflicts_rules(signals: Dict[str, Any]) -> List[str]:
        """
        Rule-based fallback for conflict detection.
        Used when LLM is unavailable or returns malformed output.
        """
        conflicts = []
        a3, a5, a6 = signals["a3"], signals["a5"], signals["a6"]

        # Conflict 1: Low risk but severe adverse media
        if a3["high_risk_flag"] == "False" and a5["is_critical"]:
            conflicts.append(
                "Agent 3 says risk is LOW/MEDIUM but Agent 5 found CRITICAL adverse media — "
                "adverse media takes precedence."
            )

        # Conflict 2: Clean adverse media but very negative sentiment
        if a5["severity"] in ("NONE", "LOW") and a6["is_very_negative"]:
            conflicts.append(
                "Agent 5 found no significant adverse news but Agent 6 reports very negative "
                "customer sentiment — possible emerging issue."
            )

        # Conflict 3: High risk classification but positive everything else
        if a3["risk_level"] and a5["severity"] in ("NONE", "LOW") and not a6["is_very_negative"]:
            conflicts.append(
                "Agent 3 classified as HIGH risk (industry-based) but adverse media and sentiment "
                "are clean — proceed with enhanced due diligence."
            )

        return conflicts

    @staticmethod
    def detect_conflicts(signals: Dict[str, Any]) -> List[str]:
        """
        LLM-powered conflict detection across Agents 3, 5, and 6.
 
        Falls back to rule-based detection if LLM is unavailable or returns
        malformed output.
        """
        llm = _get_llm()
        if llm is None:
            return OrchestrationPolicy._detect_conflicts_rules(signals)
 
        a3 = signals["a3"]
        a5 = signals["a5"]
        a6 = signals["a6"]
 
        system_prompt = """You are a risk orchestration analyst for a financial institution's
merchant onboarding pipeline. Your job is to identify CONFLICTS between three risk
assessment agents:
 
- Agent 3 (Risk & Compliance): industry-based risk classification (LOW/MEDIUM/HIGH)
- Agent 5 (Adverse Media): negative news screening (severity NONE/LOW/MEDIUM/HIGH/CRITICAL, score 0-100)
- Agent 6 (Customer Sentiment): customer review sentiment (-1.0 to 1.0)
 
A CONFLICT is when two or more agents send contradictory signals about merchant risk
that a human analyst would need to resolve. Examples:
- Agent 5 found CRITICAL adverse media but Agent 3 says LOW risk industry and Agent 6 shows positive customer sentiment
- Agent 6 shows very negative customer sentiment but Agent 5 found no adverse news 
- Agent 3 classified HIGH risk but Agent 5/Agent 6 are clean (over-classified)
- High-risk industry combined with insufficient adverse media data
 
Do NOT flag normal multi-signal merchants as conflicts. A MEDIUM-risk industry merchant
with some HIGH adverse media findings is NOT a conflict — both are warning signals
pointing the same direction.
 
Return ONLY valid JSON:
{
  "conflicts": ["short description 1", "short description 2"],
  "reasoning": "3-4 sentence explanation of why these are or aren't conflicts"
}
 
If no conflicts exist, return {"conflicts": [], "reasoning": "..."}."""
 
        user_prompt = f"""Analyze these three agent signals for conflicts:
 
Agent 3 (Risk & Compliance):
- Risk level: {a3['risk_level']}
- Is high risk: {a3['high_risk_flag']}
- Compliance review required: {a3.get('compliance_required', False)}
 
Agent 5 (Adverse Media):
- Severity: {a5['severity']}
- Risk score: {a5['risk_score']}
- Is critical (score >= 76): {a5['is_critical']}
- Is high (score >= 51): {a5['is_high']}
- Recommended action: {a5['action']}
- Data sufficiency: {'INSUFFICIENT' if a5['data_insufficient'] else 'SUFFICIENT'}
 
Agent 6 (Customer Sentiment):
- Sentiment score (-1 to 1): {a6['sentiment_score']:.2f}
- Customer satisfaction (0-5): {a6['satisfaction']:.1f}
- Flag: {a6['flag']}
- Is very negative: {a6['is_very_negative']}
- Review count: {a6['review_count']}
 
Identify conflicts and return JSON."""
 
        try:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            parsed = _extract_json(response.content)
 
            if parsed and isinstance(parsed.get("conflicts"), list):
                return [str(c) for c in parsed["conflicts"]]
 
            # LLM returned malformed JSON — fall back
            return OrchestrationPolicy._detect_conflicts_rules(signals)
 
        except Exception as exc:
            print(f"  ⚠ LLM conflict detection failed: {exc} — using rule-based fallback")
            return OrchestrationPolicy._detect_conflicts_rules(signals)

# LLM with rule-based fallback
    @staticmethod
    def _make_decision_rules(signals: Dict[str, Any], conflicts: List[str]) -> Dict[str, Any]:
        """
        Rule-based fallback for the final decision.
        Used when LLM is unavailable or returns malformed output.
        """
        a3 = signals["a3"]
        a5 = signals["a5"]
        a6 = signals["a6"]
 
        escalation_reasons: List[str] = []
        required_actions: List[str] = []
 
        # ── TIER 1: Hard REJECT conditions ──
        if a5["is_critical"]:
            escalation_reasons.append(
                f"CRITICAL adverse media finding (severity={a5['severity']}, score={a5['risk_score']})"
            )
            return {
                "decision": FinalDecision.REJECT,
                "rationale": (
                    "Critical adverse media findings disqualify this merchant from onboarding. "
                    "Policy requires automatic rejection for confirmed sanctions, fraud convictions, "
                    "or active law enforcement investigations."
                ),
                "escalation_reasons": escalation_reasons,
                "required_actions": ["Notify compliance team", "Document rejection reason"],
            }
 
        if a5["action"] == "DECLINE":
            escalation_reasons.append("Agent 5 recommended DECLINE based on adverse media analysis")
            return {
                "decision": FinalDecision.REJECT,
                "rationale": "Agent 5 adverse media screening recommended rejection.",
                "escalation_reasons": escalation_reasons,
                "required_actions": ["Notify compliance team"],
            }

        if a6["flag"] == SentimentFlag.REJECT or a6["is_very_negative"]:
            escalation_reasons.append(
                f"Very negative customer sentiment (score={a6['sentiment_score']:.2f}, "
                f"satisfaction={a6['satisfaction']:.1f}/5)"
        )
 
        # ── TIER 2: MANUAL_REVIEW conditions ──
        if a5["is_high"]:
            escalation_reasons.append(f"High-severity adverse media (score={a5['risk_score']})")
 
        if a3["high_risk_flag"]:
            escalation_reasons.append(f"High-risk industry classification ({a3.get('risk_level', 'HIGH')})")
 
        if a3["tpb_threshold_exceeded"]:
            escalation_reasons.append("Third-party business thresholds exceeded")
 
        if conflicts:
            escalation_reasons.append(f"{len(conflicts)} conflicting signals detected across agents")
 
        if a5["data_insufficient"]:
            escalation_reasons.append("Insufficient adverse media data for confident assessment")
 
        if escalation_reasons:
            required_actions = [
                "Compliance analyst review required",
                "Collect documents per Agent 4 requirements",
            ]
            if a5["data_insufficient"]:
                required_actions.append("Enhanced due diligence for thin-file entity")
            if conflicts:
                required_actions.append("Resolve conflicting agent signals before decision")
 
            return {
                "decision": FinalDecision.MANUAL_REVIEW,
                "rationale": (
                    f"Manual review required due to {len(escalation_reasons)} risk indicator(s). "
                    f"See escalation_reasons for details."
                ),
                "escalation_reasons": escalation_reasons,
                "required_actions": required_actions,
            }
 
        # ── TIER 3: APPROVE (default) ──
        return {
            "decision": FinalDecision.APPROVE,
            "rationale": (
                "All agent checks passed. Merchant meets standard onboarding criteria "
                "with no significant risk indicators."
            ),
            "escalation_reasons": [],
            "required_actions": [
                "Proceed with standard onboarding",
                "Collect documents per Agent 4 requirements",
            ],
        }

    @staticmethod
    def make_decision(signals: Dict[str, Any], conflicts: List[str]) -> Dict[str, Any]:
        """Falls back to rule-based decision if LLM is unavailable or malformed."""
        llm = _get_llm()
        if llm is None:
            return OrchestrationPolicy._make_decision_rules(signals, conflicts)
 
        a3 = signals["a3"]
        a5 = signals["a5"]
        a6 = signals["a6"]
 
        system_prompt = """You are the lead onboarding decision agent for a financial institution.
You must produce a final onboarding decision (APPROVE / MANUAL_REVIEW / REJECT) based on
signals from three risk assessment agents.
 
POLICY RULES (these are NON-NEGOTIABLE):
 
REJECT (Tier 1 — automatic rejection):
- Agent 5 severity = CRITICAL OR risk_score >= 76 (critical adverse media)
- Agent 5 recommended_action = DECLINE
- Agent 6 flag = REJECT OR sentiment_score <= -0.6 (very negative sentiment)
- Confirmed sanctions, fraud convictions, or active law enforcement investigations
 
MANUAL_REVIEW (Tier 2 — escalate to human analyst):
- Agent 5 severity = HIGH OR risk_score >= 51 (high-severity adverse media)
- Agent 3 risk_level = HIGH (high-risk industry)
- Agent 6 flag = REVIEW
- Agent 5 severity = MEDIUM OR risk_score >= 26 (medium-severity adverse media)
- Any detected conflicts between agents
- Agent 5 data_sufficiency = INSUFFICIENT (thin-file entity)
 
APPROVE (Tier 3 — default if no escalation criteria triggered):
- All agent checks passed
- No significant risk indicators
 
REJECT takes precedence over MANUAL_REVIEW. MANUAL_REVIEW takes precedence over APPROVE.
 
Return ONLY valid JSON in this exact format:
{
  "decision": "APPROVE" | "MANUAL_REVIEW" | "REJECT",
  "rationale": "2-3 sentence explanation of why this decision was made",
  "escalation_reasons": ["specific reason 1", "specific reason 2"],
  "required_actions": ["next step 1", "next step 2"]
}
 
Be precise. Reference specific signals (scores, severities, flags) in the rationale."""
 
        user_prompt = f"""Make the onboarding decision for this merchant:
 
AGENT 3 (Risk & Compliance):
- Risk level: {a3['risk_level']}
- Is high risk: {a3['high_risk_flag']}
- TPB threshold exceeded: {a3.get('tpb_threshold_exceeded', False)}
- Compliance review required: {a3.get('compliance_required', False)}
 
AGENT 5 (Adverse Media):
- Severity: {a5['severity']}
- Risk score: {a5['risk_score']} / 100
- Is critical: {a5['is_critical']}
- Is high: {a5['is_high']}
- Is medium: {a5['is_medium']}
- Recommended action: {a5['action']}
- Data sufficiency: {'INSUFFICIENT' if a5['data_insufficient'] else 'SUFFICIENT'}
 
AGENT 6 (Customer Sentiment):
- Sentiment score: {a6['sentiment_score']:.2f}
- Customer satisfaction: {a6['satisfaction']:.1f} / 5
- Flag: {a6['flag']}
- Is very negative: {a6['is_very_negative']}
- Review count: {a6['review_count']}
 
CONFLICTS DETECTED ({len(conflicts)}):
{json.dumps(conflicts, indent=2) if conflicts else 'None'}
 
Apply the policy rules and return your decision as JSON."""
 
        try:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            parsed = _extract_json(response.content)
 
            if not parsed:
                print("  ⚠ LLM decision returned malformed JSON — using rule-based fallback")
                return OrchestrationPolicy._make_decision_rules(signals, conflicts)
 
            decision_str = str(parsed.get("decision", "")).upper()
            if decision_str not in ("APPROVE", "MANUAL_REVIEW", "REJECT"):
                print(f"  ⚠ LLM returned invalid decision '{decision_str}' — using fallback")
                return OrchestrationPolicy._make_decision_rules(signals, conflicts)
 
            decision_map = {
                "APPROVE": FinalDecision.APPROVE,
                "MANUAL_REVIEW": FinalDecision.MANUAL_REVIEW,
                "REJECT": FinalDecision.REJECT,
            }
 
            return {
                "decision": decision_map[decision_str],
                "rationale": str(parsed.get("rationale", "LLM decision (no rationale provided)")),
                "escalation_reasons": [
                    str(r) for r in parsed.get("escalation_reasons", [])
                    if isinstance(r, (str, int, float))
                ],
                "required_actions": [
                    str(a) for a in parsed.get("required_actions", [])
                    if isinstance(a, (str, int, float))
                ],
            }
 
        except Exception as exc:
            print(f"  ⚠ LLM decision making failed: {exc} — using rule-based fallback")
            return OrchestrationPolicy._make_decision_rules(signals, conflicts)
     
# LANGGRAPH NODES
def node_intake(state: OrchestrationState) -> Dict[str, Any]:
    """Node 1: INTAKE — Validate that all required agent outputs are present."""
    trace = state.get("processing_trace", [])
    trace.append(f"[{_now()}] INTAKE: Receiving outputs from Agents 3, 4, 5, 6")
 
    missing = []
    for agent_key in ("agent3", "agent4", "agent5", "agent6"):
        if not state.get(agent_key):
            missing.append(agent_key)
 
    if missing:
        trace.append(f"[{_now()}] INTAKE: ⚠️ Missing inputs: {missing}")
    else:
        trace.append(f"[{_now()}] INTAKE: ✓ All 4 agent inputs received")
 
    orch_id = (
        f"ORCH-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-"
        f"{hashlib.md5(state['merchant_name'].encode()).hexdigest()[:8]}"
    )
 
    return {
        "orchestration_id": orch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_trace": trace,
    }
 
 
def node_assess(state: OrchestrationState) -> Dict[str, Any]:
    """Node 2: ASSESS — Extract normalized signals from each agent output."""
    trace = state["processing_trace"]
    trace.append(f"[{_now()}] ASSESS: Extracting signals from each agent")
 
    signals = {
        "a3": OrchestrationPolicy.assess_agent3(state["agent3"]),
        "a4": OrchestrationPolicy.assess_agent4(state["agent4"]),
        "a5": OrchestrationPolicy.assess_agent5(state["agent5"]),
        "a6": OrchestrationPolicy.assess_agent6(state["agent6"]),
    }
 
    trace.append(
        f"[{_now()}] ASSESS: A3 risk={signals['a3']['risk_level']} | "
        f"A4 type={signals['a4'].get('merchant_type', 'N/A')} pd={signals['a4']['pd_score']} | "
        f"A5 severity={signals['a5']['severity']} score={signals['a5']['risk_score']} | "
        f"A6 sentiment={signals['a6']['sentiment_score']:.2f}"
    )
 
    return {"signals": signals, "processing_trace": trace}
 
 
def node_conflict_detection(state: OrchestrationState) -> Dict[str, Any]:
    """Node 3: CONFLICT — LLM detects contradictions between agent signals."""
    trace = state["processing_trace"]
    trace.append(f"[{_now()}] CONFLICT: Asking LLM to detect conflicts (Agents 3/5/6)")
 
    conflicts = OrchestrationPolicy.detect_conflicts(state["signals"])
    has_critical = any("CRITICAL" in c or "precedence" in c for c in conflicts)
 
    if conflicts:
        trace.append(f"[{_now()}] CONFLICT: ⚠️ {len(conflicts)} conflict(s) detected")
        for c in conflicts:
            trace.append(f"[{_now()}]   → {c}")
    else:
        trace.append(f"[{_now()}] CONFLICT: ✓ No conflicts between agent signals")
 
    return {
        "conflicts": conflicts,
        "has_critical_conflict": has_critical,
        "processing_trace": trace,
    }
 
 
def node_decide(state: OrchestrationState) -> Dict[str, Any]:
    """Node 4: DECIDE — LLM applies policy rules to produce the final decision."""
    trace = state["processing_trace"]
    trace.append(f"[{_now()}] DECIDE: Asking LLM to make final decision")
 
    result = OrchestrationPolicy.make_decision(state["signals"], state["conflicts"])
 
    trace.append(
        f"[{_now()}] DECIDE: → {result['decision']} "
        f"({len(result['escalation_reasons'])} escalation reason(s))"
    )
 
    return {
        "final_decision": result["decision"],
        "escalation_reasons": result["escalation_reasons"],
        "decision_rationale": result["rationale"],
        "required_actions": result["required_actions"],
        "processing_trace": trace,
    }
 
 
def node_finalize(state: OrchestrationState) -> Dict[str, Any]:
    """Node 5: FINALIZE — Produce the final audit-ready output."""
    trace = state["processing_trace"]
    trace.append(f"[{_now()}] FINALIZE: Decision = {state['final_decision']}")
    trace.append(f"[{_now()}] FINALIZE: Orchestration complete")
    return {"processing_trace": trace}

# EDGE ROUTING & GRAPH BUILDER
def route_after_intake(state: OrchestrationState) -> Literal["assess", "finalize"]:
    """If any agent output is missing, skip to finalize with MANUAL_REVIEW."""
    required = ["agent3", "agent4", "agent5", "agent6"]
    if any(not state.get(k) for k in required):
        return "finalize"
    return "assess"
 
 
def build_orchestration_graph():

    graph = StateGraph(OrchestrationState)
 
    graph.add_node("intake", node_intake)
    graph.add_node("assess", node_assess)
    graph.add_node("conflict", node_conflict_detection)
    graph.add_node("decide", node_decide)
    graph.add_node("finalize", node_finalize)
 
    graph.set_entry_point("intake")
 
    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {"assess": "assess", "finalize": "finalize"},
    )
 
    graph.add_edge("assess", "conflict")
    graph.add_edge("conflict", "decide")
    graph.add_edge("decide", "finalize")
    graph.add_edge("finalize", END)
 
    return graph.compile()

#  AGENT 7 WRAPPER
class Agent7OrchestrationAgent:
    """
    Orchestration Agent — coordinates Agents 3, 4, 5, 6 into a unified decision.
 
    Usage:
        agent = Agent7OrchestrationAgent()
        result = agent.process(
            merchant_name="PayPal",
            agent3_output=...,
            agent4_output=...,
            agent5_output=...,
            agent6_output=...,
        )
    """
 
    def __init__(self):
        self._graph = build_orchestration_graph()
 
    def process(
        self,
        merchant_name: str,
        agent3_output: Agent3Output,
        agent4_output: Agent4Output,
        agent5_output: Agent5Output,
        agent6_output: Agent6Output,
    ) -> Dict[str, Any]:
        """Run the orchestration graph and return the final decision."""
        initial_state: OrchestrationState = {
            "merchant_name": merchant_name,
            "agent3": asdict(agent3_output),
            "agent4": asdict(agent4_output),
            "agent5": asdict(agent5_output),
            "agent6": asdict(agent6_output),
            "signals": {},
            "conflicts": [],
            "has_critical_conflict": False,
            "final_decision": "",
            "escalation_reasons": [],
            "decision_rationale": "",
            "required_actions": [],
            "orchestration_id": "",
            "timestamp": "",
            "processing_trace": [],
        }
 
        final_state = self._graph.invoke(initial_state)
        self._print_report(final_state)
        return final_state
 
    def _print_report(self, state: Dict[str, Any]):
        """Print the orchestration result to console."""
        decision_icon = {
            "APPROVE": "✅",
            "MANUAL_REVIEW": "⚠️",
            "REJECT": "🚫",
        }.get(state["final_decision"], "?")
 
        print(f"\n{'='*70}")
        print(f"  AGENT 7 — ORCHESTRATION DECISION: {state['merchant_name']}")
        print(f"{'='*70}")
        print(f"  Orchestration ID: {state['orchestration_id']}")
        print(f"  Timestamp:        {state['timestamp']}")
        print(f"  Decision:         {decision_icon} {state['final_decision']}")
        print(f"\n  RATIONALE:")
        print(f"    {state['decision_rationale']}")
 
        if state.get("escalation_reasons"):
            print(f"\n  ESCALATION REASONS ({len(state['escalation_reasons'])}):")
            for i, reason in enumerate(state["escalation_reasons"], 1):
                print(f"    {i}. {reason}")
 
        if state.get("conflicts"):
            print(f"\n  CONFLICTS DETECTED ({len(state['conflicts'])}):")
            for c in state["conflicts"]:
                print(f"    • {c}")
 
        if state.get("required_actions"):
            print(f"\n  REQUIRED ACTIONS:")
            for action in state["required_actions"]:
                print(f"    → {action}")
 
        print(f"\n  PROCESSING TRACE:")
        for step in state["processing_trace"]:
            print(f"    {step}")
        print(f"{'='*70}\n")
 
 
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")
 
# Example
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
 
    orchestrator = Agent7OrchestrationAgent()
 
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Clean merchant (expect APPROVE)")
    print("=" * 70)
 
    result1 = orchestrator.process(
        merchant_name="PayPal",
        agent3_output=Agent3Output(
            merchant_name="PayPal",
            risk_level="MEDIUM",
            industry_risk_category="Fintech",
            compliance_review_required=False,
            tpb_threshold_exceeded=False,
        ),
        agent4_output=Agent4Output(
            merchant_name="PayPal",
            merchant_type="Existing Merchant",
            risk_review="Light Review",
            pd_score=0.15,
            documents_required=["bank_statement"],
            merchant_industry="Fintech",
            high_risk_match=False,
            compliance_message="Not high risk",
        ),
        agent5_output=Agent5Output(
            merchant_name="PayPal",
            overall_risk_severity="LOW",
            overall_risk_score=12.0,
            recommended_action="APPROVE",
            data_sufficiency="SUFFICIENT",
            findings_count=1,
            corroboration_score=0.3,
        ),
        agent6_output=Agent6Output(
            merchant_name="PayPal",
            overall_sentiment_score=0.25,
            customer_satisfaction_rating=3.8,
            recommendation_flag="APPROVE",
            review_count=45,
            negative_ratio=0.15,
        ),
    )
 
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Critical adverse media (expect REJECT)")
    print("=" * 70)
 
    result2 = orchestrator.process(
        merchant_name="ShadyPayments Inc",
        agent3_output=Agent3Output(
            merchant_name="ShadyPayments Inc",
            risk_level="LOW",
            industry_risk_category="Payments",
            tpb_threshold_exceeded=True,
        ),
        agent4_output=Agent4Output(
            merchant_name="ShadyPayments Inc",
            merchant_type="New Merchant",
            risk_review="Full Review",
            documents_required=["incorporation_docs", "bank_statement", "id_proof"],
            merchant_industry="Payments",
        ),
        agent5_output=Agent5Output(
            merchant_name="ShadyPayments Inc",
            overall_risk_severity="CRITICAL",
            overall_risk_score=85.0,
            recommended_action="DECLINE",
            data_sufficiency="SUFFICIENT",
            findings_count=4,
            corroboration_score=0.85,
        ),
        agent6_output=Agent6Output(
            merchant_name="ShadyPayments Inc",
            overall_sentiment_score=-0.8,
            customer_satisfaction_rating=0.9,
            recommendation_flag="REJECT",
            review_count=22,
            negative_ratio=0.85,
        ),
    )
 
    with open("agent7_orchestration_results.json", "w") as f:
        json.dump({
            "paypal": result1,
            "shady_payments": result2,
        }, f, indent=2, default=str)
    print("\n📁 Saved: agent7_orchestration_results.json")
