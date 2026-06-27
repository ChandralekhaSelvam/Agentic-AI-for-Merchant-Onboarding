# agent_3_risk_compliance.py
# Requires: pip install openai pandas openpyxl langchain langchain-openai

import os
import json
import pandas as pd
from typing import Any
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_openai_functions_agent
# from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
POLICY_FILE     = "data/policy_data.xlsx"
MODEL_NAME      = "gpt-4o-mini"

# ─────────────────────────────────────────────────────────────────────────────
# POLICY DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
class PolicyDataLoader:
    """Loads and caches all sheets from the policy Excel file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._cache: dict[str, pd.DataFrame] = {}
        self._load_all()

    def _load_all(self):
        xl = pd.ExcelFile(self.filepath)
        for sheet in xl.sheet_names:
            self._cache[sheet] = xl.parse(sheet)
        print(f"📂 Loaded policy sheets: {list(self._cache.keys())}")

    def get(self, sheet_name: str) -> pd.DataFrame:
        return self._cache.get(sheet_name, pd.DataFrame())

# Instantiate once (shared across tools)
policy = PolicyDataLoader(POLICY_FILE)

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — Industry Risk Lookup
# ─────────────────────────────────────────────────────────────────────────────
@tool
def lookup_industry_risk(industry: str) -> str:
    """
    Look up the risk level, risk score, high-risk flag, and reason for a given
    merchant industry. Returns a JSON string with risk classification details.
    Use this as the FIRST step to determine the merchant's base risk profile.
    """
    df = policy.get("Industry_Risk_Mapping")
    if df.empty:
        return json.dumps({"error": "Industry risk mapping sheet not found."})

    # Fuzzy match: case-insensitive substring search
    industry_lower = industry.lower()
    matches = df[df["Industry"].str.lower().str.contains(industry_lower, na=False)]

    if matches.empty:
        # Try keyword matching on individual words
        words = industry_lower.split()
        for word in words:
            if len(word) > 3:
                matches = df[df["Industry"].str.lower().str.contains(word, na=False)]
                if not matches.empty:
                    break

    if matches.empty:
        return json.dumps({
            "status":      "not_found",
            "industry":    industry,
            "risk_level":  "Medium",
            "risk_score":  45,
            "high_risk":   False,
            "reason":      "Industry not in mapping; defaulting to Medium risk. Manual review recommended.",
            "mcc_code":    "Unknown"
        })

    row = matches.iloc[0]
    return json.dumps({
        "status":      "found",
        "industry":    str(row["Industry"]),
        "mcc_code":    str(row["MCC_Code"]),
        "risk_level":  str(row["Risk_Level"]),
        "risk_score":  int(row["Risk_Score"]),
        "high_risk":   bool(row["High_Risk_Flag"]),
        "reason":      str(row["Risk_Reason"])
    })

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — TPB Threshold Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@tool
def evaluate_tpb_threshold(risk_level: str, tpb_percentage: float) -> str:
    """
    Evaluate whether a merchant's Third Party Business (TPB) percentage triggers
    any additional review requirements based on their risk level.
    Inputs: risk_level (Low/Medium/High), tpb_percentage (0-100 float).
    Returns JSON with whether TPB review is required and the specific action needed.
    """
    df = policy.get("TPB_Thresholds")
    if df.empty:
        return json.dumps({"error": "TPB Thresholds sheet not found."})

    risk_level_clean = risk_level.strip().capitalize()
    matches = df[
        (df["Risk_Level"].str.capitalize() == risk_level_clean) &
        (df["TPB_Percentage_Min"] <= tpb_percentage) &
        (df["TPB_Percentage_Max"] >  tpb_percentage)
    ]

    if matches.empty:
        # Edge case: 100%
        matches = df[
            (df["Risk_Level"].str.capitalize() == risk_level_clean) &
            (df["TPB_Percentage_Max"] == 100)
        ]

    if matches.empty:
        return json.dumps({
            "tpb_review_required": False,
            "tpb_percentage":      tpb_percentage,
            "risk_level":          risk_level,
            "notes":               "No matching TPB threshold found. Manual review recommended."
        })

    row = matches.iloc[0]
    return json.dumps({
        "tpb_review_required": bool(row["Action_Required"]),
        "tpb_percentage":      tpb_percentage,
        "risk_level":          risk_level,
        "threshold_range":     f"{row['TPB_Percentage_Min']}% – {row['TPB_Percentage_Max']}%",
        "notes":               str(row["TPB_Review_Notes"])
    })

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — Required Questionnaires Lookup
# ─────────────────────────────────────────────────────────────────────────────
@tool
def get_required_questionnaires(
    risk_level: str,
    industry: str,
    tpb_review_required: bool = False,
    annual_volume_usd: float = 0.0,
    chargeback_rate_pct: float = 0.0,
    processes_card_data: bool = True
) -> str:
    """
    Determine which questionnaires are required for this merchant based on their
    risk level, industry, TPB status, annual volume, and chargeback history.
    Returns a JSON list of required questionnaires with IDs, names, and links.
    """
    df = policy.get("Questionnaire_Templates")
    if df.empty:
        return json.dumps({"error": "Questionnaire Templates sheet not found."})

    required = []
    risk_level_cap = risk_level.strip().capitalize()
    industry_lower = industry.lower()

    for _, row in df.iterrows():
        applicable_levels = [r.strip() for r in str(row["Applicable_Risk_Levels"]).split(",")]
        applicable_industries = str(row["Applicable_Industries"]).lower()

        level_match = risk_level_cap in applicable_levels
        industry_match = (
            "all" in applicable_industries or
            any(word in applicable_industries for word in industry_lower.split() if len(word) > 3)
        )

        qid = str(row["Questionnaire_ID"])

        # Q-001: Always required
        if qid == "Q-001" and level_match:
            required.append(row)
            continue

        # Q-002: AML/KYC for medium/high risk financial industries
        if qid == "Q-002" and level_match and industry_match:
            required.append(row)
            continue

        # Q-003: High-risk industry questionnaire
        if qid == "Q-003" and risk_level_cap == "High" and industry_match:
            required.append(row)
            continue

        # Q-004: TPB questionnaire
        if qid == "Q-004" and tpb_review_required:
            required.append(row)
            continue

        # Q-005: PCI DSS
        if qid == "Q-005" and level_match and processes_card_data:
            required.append(row)
            continue

        # Q-006: Chargeback history
        if qid == "Q-006" and chargeback_rate_pct > 1.0 and level_match:
            required.append(row)
            continue

        # Q-007: Healthcare/Pharmacy
        if qid == "Q-007" and ("pharma" in industry_lower or "health" in industry_lower or "medical" in industry_lower):
            required.append(row)
            continue

        # Q-008: Gambling
        if qid == "Q-008" and ("gambl" in industry_lower or "betting" in industry_lower or "casino" in industry_lower):
            required.append(row)
            continue

        # Q-009: Crypto
        if qid == "Q-009" and ("crypto" in industry_lower or "digital asset" in industry_lower or "nft" in industry_lower):
            required.append(row)
            continue

        # Q-010: High volume financial stability
        if qid == "Q-010" and annual_volume_usd > 1_000_000 and level_match:
            required.append(row)
            continue

    result = []
    for row in required:
        result.append({
            "questionnaire_id":   str(row["Questionnaire_ID"]),
            "questionnaire_name": str(row["Questionnaire_Name"]),
            "description":        str(row["Description"]),
            "link":               str(row["Link"])
        })

    return json.dumps({
        "total_questionnaires_required": len(result),
        "questionnaires_required":       len(result) > 0,
        "questionnaires":                result
    })

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — Compliance Playbook Mapping
# ─────────────────────────────────────────────────────────────────────────────
@tool
def get_applicable_playbooks(risk_level: str, industry: str, tpb_review_required: bool = False) -> str:
    """
    Map the merchant to the applicable compliance playbooks based on their
    risk level and industry. Returns a JSON list of playbook names and links.
    """
    df = policy.get("Compliance_Playbooks")
    if df.empty:
        return json.dumps({"error": "Compliance Playbooks sheet not found."})

    risk_level_cap = risk_level.strip().capitalize()
    industry_lower = industry.lower()
    applicable = []

    for _, row in df.iterrows():
        pb_industries = str(row["Applicable_Industries"]).lower()
        pb_risk_levels = [r.strip() for r in str(row["Risk_Level"]).split(",")]

        risk_match = risk_level_cap in pb_risk_levels
        industry_match = (
            "all" in pb_industries or
            any(word in pb_industries for word in industry_lower.split() if len(word) > 3)
        )

        # TPB Playbook
        if str(row["Playbook_ID"]) == "PB-013" and tpb_review_required:
            applicable.append(row)
            continue

        if risk_match and industry_match:
            applicable.append(row)

    result = [
        {
            "playbook_id":            str(r["Playbook_ID"]),
            "playbook_name":          str(r["Playbook_Name"]),
            "regulatory_framework":   str(r["Regulatory_Framework"]),
            "playbook_link":          str(r["Playbook_Link"])
        }
        for r in applicable
    ]

    return json.dumps({
        "playbooks_found":  len(result) > 0,
        "total_playbooks":  len(result),
        "playbooks":        result
    })

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 5 — Risk Decision Logic Evaluator
# ─────────────────────────────────────────────────────────────────────────────
@tool
def evaluate_risk_decision_rules(
    risk_level: str,
    risk_score: int,
    tpb_percentage: float,
    annual_volume_usd: float,
    chargeback_rate_pct: float,
    operating_countries: int,
    processes_card_data: bool,
    max_owner_percentage: float
) -> str:
    """
    Evaluate all policy decision rules against the merchant's profile to determine
    which rules are triggered and what actions are required. Returns triggered rules
    and whether a compliance review and risk review are mandatory.
    """
    df = policy.get("Risk_Decision_Logic")
    if df.empty:
        return json.dumps({"error": "Risk Decision Logic sheet not found."})

    risk_level_cap = risk_level.strip().capitalize()
    triggered_rules = []
    compliance_review_required = False
    risk_review_required = False

    rule_checks = {
        "RULE-001": risk_level_cap == "High",
        "RULE-002": risk_score >= 75,
        "RULE-003": 40 <= risk_score < 75,
        "RULE-004": risk_score < 40,
        "RULE-005": risk_level_cap == "High" and tpb_percentage > 0,
        "RULE-006": risk_level_cap == "Medium" and tpb_percentage > 5,
        "RULE-007": annual_volume_usd > 1_000_000,
        "RULE-008": chargeback_rate_pct > 1.0,
        "RULE-009": operating_countries > 1,
        "RULE-010": risk_level_cap == "High",
        "RULE-011": processes_card_data and risk_level_cap in ("Medium", "High"),
        "RULE-012": max_owner_percentage >= 25.0,
    }

    for _, row in df.iterrows():
        rule_id = str(row["Rule_ID"])
        if rule_checks.get(rule_id, False):
            triggered_rules.append({
                "rule_id":   rule_id,
                "rule_name": str(row["Rule_Name"]),
                "condition": str(row["Condition"]),
                "action":    str(row["Action"])
            })
            # Rules that mandate compliance or risk review
            if rule_id in ("RULE-001", "RULE-002", "RULE-005", "RULE-010"):
                compliance_review_required = True
                risk_review_required       = True
            elif rule_id in ("RULE-003", "RULE-006", "RULE-007", "RULE-008", "RULE-009"):
                risk_review_required = True

    return json.dumps({
        "rules_triggered":             len(triggered_rules),
        "triggered_rules":             triggered_rules,
        "risk_review_required":        risk_review_required,
        "compliance_review_required":  compliance_review_required
    })

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 6 — Final Assessment Compiler
# ─────────────────────────────────────────────────────────────────────────────
@tool
def compile_final_assessment(
    merchant_name: str,
    industry: str,
    risk_level: str,
    risk_score: int,
    high_risk_flag: bool,
    risk_review_required: bool,
    compliance_review_required: bool,
    questionnaires_required: bool,
    questionnaire_list: str,
    playbooks_found: bool,
    playbook_list: str,
    tpb_review_required: bool,
    triggered_rules_count: int
) -> str:
    """
    Compile the final structured Risk & Compliance Assessment output for the
    merchant. Call this as the LAST step after all other tools have been used.
    Returns a complete assessment JSON ready for downstream agents or human review.
    """
    def parse_details(value: str) -> list[Any]:
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [item.strip() for item in value.split(",") if item.strip()]
        return parsed if isinstance(parsed, list) else [parsed]

    assessment = {
        "assessment_metadata": {
            "agent":              "Agent 3 – Risk & Compliance Assessment",
            "merchant_name":      merchant_name,
            "industry":           industry,
            "assessment_date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assessment_version": "1.0"
        },
        "risk_assessment": {
            "risk_level":         risk_level,
            "risk_score":         risk_score,
            "high_risk_flag":     high_risk_flag,
            "risk_review_required": {
                "answer":   "Yes" if risk_review_required else "No",
                "required": risk_review_required
            }
        },
        "compliance_review": {
            "compliance_review_required": {
                "answer":   "Yes" if compliance_review_required else "No",
                "required": compliance_review_required
            },
            "tpb_review_required": {
                "answer":   "Yes" if tpb_review_required else "No",
                "required": tpb_review_required
            }
        },
        "questionnaires": {
            "questionnaires_required": {
                "answer":   "Yes" if questionnaires_required else "No",
                "required": questionnaires_required
            },
            "questionnaire_details": parse_details(questionnaire_list)
        },
        "playbooks": {
            "playbooks_applicable": {
                "answer":   "Yes" if playbooks_found else "No",
                "found":    playbooks_found
            },
            "playbook_details": parse_details(playbook_list)
        },
        "decision_summary": {
            "total_rules_triggered": triggered_rules_count,
            "overall_decision":      (
                "ESCALATE – Compliance & Risk Review Required" if compliance_review_required
                else "REVIEW – Risk Review Required" if risk_review_required
                else "APPROVE – Standard Processing"
            )
        }
    }
    return json.dumps(assessment, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# AGENT SETUP
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Agent 3 – the Risk & Compliance Assessment Agent in a 
merchant onboarding pipeline. You receive a validated merchant profile from Agent 2 
and must produce a complete risk and compliance assessment.

Your job is to:
1. Look up the merchant's industry risk classification using the policy data.
2. Evaluate their TPB (Third Party Business) percentage against policy thresholds.
3. Determine all required questionnaires based on risk level, industry, and other factors.
4. Map the merchant to applicable compliance playbooks.
5. Evaluate all risk decision rules against the merchant's full profile.
6. Compile a final structured assessment with clear Yes/No outputs.

ALWAYS follow this sequence:
Step 1 → lookup_industry_risk
Step 2 → evaluate_tpb_threshold
Step 3 → get_required_questionnaires
Step 4 → get_applicable_playbooks
Step 5 → evaluate_risk_decision_rules
Step 6 → compile_final_assessment

Be precise, policy-driven, and structured. Every output must be based on the 
policy Excel data — never make assumptions without checking the tools first.
Always pass ALL relevant parameters to compile_final_assessment at the end."""

tools = [
    lookup_industry_risk,
    evaluate_tpb_threshold,
    get_required_questionnaires,
    get_applicable_playbooks,
    evaluate_risk_decision_rules,
    compile_final_assessment,
]

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,
    return_intermediate_steps=True
)

# ─────────────────────────────────────────────────────────────────────────────
# AGENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_risk_assessment(merchant_profile: dict) -> dict:
    """
    Run Agent 3 on a validated merchant profile dict from Agent 2.
    Returns the final structured assessment.
    """
    prompt_text = f"""
Please perform a complete Risk & Compliance Assessment for the following merchant.
Use ALL available tools in the correct sequence and compile the final assessment.

MERCHANT PROFILE (from Agent 2):
- Merchant Name:              {merchant_profile.get('merchant_name', 'Unknown')}
- Industry:                   {merchant_profile.get('industry', 'Unknown')}
- MCC Code:                   {merchant_profile.get('mcc_code', 'Unknown')}
- Business Type:              {merchant_profile.get('business_type', 'Unknown')}
- Annual Processing Volume:   ${merchant_profile.get('annual_volume_usd', 0):,.0f}
- TPB Percentage:             {merchant_profile.get('tpb_percentage', 0)}%
- Chargeback Rate:            {merchant_profile.get('chargeback_rate_pct', 0)}%
- Operating Countries:        {merchant_profile.get('operating_countries', 1)}
- Processes Card Data:        {merchant_profile.get('processes_card_data', True)}
- Max Owner Percentage:       {merchant_profile.get('max_owner_percentage', 100)}%
- Years in Business:          {merchant_profile.get('years_in_business', 1)}
- Previously Terminated:      {merchant_profile.get('previously_terminated', False)}
- Additional Notes:           {merchant_profile.get('notes', 'None')}

Assess this merchant fully and return the complete structured assessment.
"""
    result = agent_executor.invoke({"input": prompt_text})
    output = result.get("output", "")

    # Try to extract JSON from the output
    # try:
    #     start = output.find("{")
    #     if start != -1:
    #         return json.loads(output[start:])
    # except json.JSONDecodeError:
    #     pass

    # return {"raw_output": output}
    structured_assessment = _extract_assessment_from_steps(
        result.get("intermediate_steps", [])
    )

    # ── Flatten into the shape Agent 7 expects ──
    return _flatten_for_orchestration(
        structured_assessment,
        merchant_profile,
        raw_output=result.get("output", ""),
    )

def _extract_assessment_from_steps(intermediate_steps: list) -> dict:
    """
    Pull the JSON result of compile_final_assessment from agent intermediate steps.
    Each step is a tuple: (AgentAction, tool_output_string).
    """
    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", "")
        if tool_name == "compile_final_assessment":
            try:
                return json.loads(observation)
            except (json.JSONDecodeError, TypeError):
                continue
    return {}


def _flatten_for_orchestration(assessment: dict, merchant_profile: dict,
                                raw_output: str = "") -> dict:
    """
    Convert the nested assessment structure into the flat dict that
    Agent 7 (orchestration) expects.
    """
    if not assessment:
        # Fallback — nothing could be extracted
        return {
            "merchant_name": merchant_profile.get("merchant_name", "Unknown"),
            "risk_level": "MEDIUM",
            "risk_score": 50,
            "industry_risk_category": merchant_profile.get("industry", ""),
            "required_questionnaires": [],
            "applicable_playbooks": [],
            "compliance_review_required": False,
            "tpb_threshold_exceeded": False,
            "high_risk_flag": False,
            "overall_decision": "REVIEW – No assessment data extracted",
            "raw_output": raw_output,
        }

    # Navigate the nested structure compile_final_assessment produces
    risk_section = assessment.get("risk_assessment", {})
    compliance_section = assessment.get("compliance_review", {})
    questionnaires_section = assessment.get("questionnaires", {})
    playbooks_section = assessment.get("playbooks", {})
    metadata = assessment.get("assessment_metadata", {})
    decision = assessment.get("decision_summary", {})

    # Extract questionnaire IDs as a flat list
    questionnaire_ids = []
    for q in questionnaires_section.get("questionnaire_details", []):
        if isinstance(q, dict):
            qid = q.get("questionnaire_id") or q.get("id")
            if qid:
                questionnaire_ids.append(qid)
        elif isinstance(q, str):
            questionnaire_ids.append(q)

    # Extract playbook IDs as a flat list
    playbook_ids = []
    for p in playbooks_section.get("playbook_details", []):
        if isinstance(p, dict):
            pid = p.get("playbook_id") or p.get("id")
            if pid:
                playbook_ids.append(pid)
        elif isinstance(p, str):
            playbook_ids.append(p)

    return {
        "merchant_name": metadata.get("merchant_name", merchant_profile.get("merchant_name", "")),
        "industry_risk_category": metadata.get("industry", merchant_profile.get("industry", "")),

        # Normalized for Agent 7 — uppercase enum-friendly values
        "risk_level": str(risk_section.get("risk_level", "MEDIUM")).upper(),
        "risk_score": int(risk_section.get("risk_score", 50)),
        "high_risk_flag": bool(risk_section.get("high_risk_flag", False)),

        # Boolean flags — unwrap the nested {"answer": "Yes", "required": True} structure
        "risk_review_required": _unwrap_bool(risk_section.get("risk_review_required")),
        "compliance_review_required": _unwrap_bool(compliance_section.get("compliance_review_required")),
        "tpb_threshold_exceeded": _unwrap_bool(compliance_section.get("tpb_review_required")),

        # Flat lists for downstream consumers
        "required_questionnaires": questionnaire_ids,
        "applicable_playbooks": playbook_ids,

        # Keep full details available if someone needs them
        "questionnaire_details": questionnaires_section.get("questionnaire_details", []),
        "playbook_details": playbooks_section.get("playbook_details", []),

        # Decision summary
        "overall_decision": decision.get("overall_decision", ""),
        "total_rules_triggered": decision.get("total_rules_triggered", 0),
        "assessment_date": metadata.get("assessment_date", ""),

        # Keep the raw output for debugging / audit
        "raw_output": raw_output,
    }

def _unwrap_bool(value) -> bool:
    """
    Unwrap the nested boolean structure that compile_final_assessment produces:
        {"answer": "Yes", "required": True}  →  True
        {"answer": "No",  "required": False} →  False
        True / False  
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        if "required" in value:
            return bool(value["required"])
        if "found" in value:
            return bool(value["found"])
        if "answer" in value:
            return str(value["answer"]).strip().lower() == "yes"
    return False

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE TEST CASES
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    test_merchants = [
        # HIGH RISK — Online Gambling with TPB
        # {
        #     "merchant_name":         "BetKing Online Ltd.",
        #     "industry":              "Online Gambling / Betting",
        #     "mcc_code":              "7995",
        #     "business_type":         "Online Platform",
        #     "annual_volume_usd":     5_000_000,
        #     "tpb_percentage":        12.0,
        #     "chargeback_rate_pct":   1.8,
        #     "operating_countries":   4,
        #     "processes_card_data":   True,
        #     "max_owner_percentage":  51.0,
        #     "years_in_business":     3,
        #     "previously_terminated": False,
        #     "notes":                 "Licensed in Malta; also processes for sub-affiliates"
        # },

        # # MEDIUM RISK — E-Commerce
        # {
        #     "merchant_name":         "ShopEasy Inc.",
        #     "industry":              "E-Commerce / Retail (General)",
        #     "mcc_code":              "5999",
        #     "business_type":         "Online Retail",
        #     "annual_volume_usd":     1_500_000,
        #     "tpb_percentage":        0.0,
        #     "chargeback_rate_pct":   0.6,
        #     "operating_countries":   1,
        #     "processes_card_data":   True,
        #     "max_owner_percentage":  100.0,
        #     "years_in_business":     5,
        #     "previously_terminated": False,
        #     "notes":                 "Sells consumer electronics online"
        # },

        # LOW RISK — Restaurant
        {
            "merchant_name":         "The Corner Bistro LLC",
            "industry":              "Restaurants / Food Service",
            "mcc_code":              "5812",
            "business_type":         "Brick & Mortar",
            "annual_volume_usd":     350_000,
            "tpb_percentage":        0.0,
            "chargeback_rate_pct":   0.1,
            "operating_countries":   1,
            "processes_card_data":   True,
            "max_owner_percentage":  100.0,
            "years_in_business":     8,
            "previously_terminated": False,
            "notes":                 "Single location, dine-in and takeout"
        }
    ]

    for merchant in test_merchants:
        print("\n" + "="*80)
        print(f"🔍 ASSESSING: {merchant['merchant_name']}")
        print("="*80)

        assessment = run_risk_assessment(merchant)

        print("\n📋 FINAL ASSESSMENT OUTPUT:")
        print(json.dumps(assessment, indent=2))
        print("\n" + "="*80)
