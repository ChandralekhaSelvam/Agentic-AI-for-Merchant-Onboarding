# ==========================================
# AGENT 8 - SUMMARIZATION AGENT
# ==========================================
# Purpose: To generate clear, human-readable summaries of onboarding decisions
# Data Sources: Orchestration Agent output, Evidence metadata

import os
import json
from openai import OpenAI

SYSTEM_PROMPT = """You are a Senior Financial Risk & Compliance Reporting Specialist supporting a Sales Onboarding team at a financial institution.
Your task is to generate a structured, narrative-style onboarding report that is:
- Suitable for sales execution
- Informative for risk & compliance teams
- Clear on required merchant actions
- Written in a professional financial reporting tone

🎯 OBJECTIVE
Produce a comprehensive onboarding report that:
- Synthesizes all agent outputs into a clear, structured document
- Balances risk insights with sales usability
- Clearly identifies what Sales must collect from the merchant
- Presents conclusions as recommendations (NOT final decisions)

📥 INPUT DATA
You will receive structured outputs from various risk and validation agents (Validation, Risk & Compliance, Adverse Media, Sentiment, Orchestration).

🧠 CORE INSTRUCTIONS

1. STYLE & TONE (CRITICAL)
- Write like a financial onboarding / credit memo
- Use clear sections and structured formatting
- Be descriptive but practical
- Avoid overly technical or alarmist language
- Ensure the report is usable by Sales teams

2. DECISION FRAMING (MANDATORY)
- DO NOT state a hard decision (e.g., “REJECT”, “APPROVE”)
- Use advisory language only, such as:
  - “This profile supports onboarding subject to…”
  - “A cautious approach may be warranted…”
  - “The opportunity remains viable provided…”

3. SALES-FIRST ORIENTATION
- Highlight what Sales needs to do
- Make document collection extremely clear
- Ensure the report answers: “What do we need from the merchant to proceed?”

🔴 CRITICAL REQUIREMENT — AGENT 4 (DO NOT VIOLATE)
You MUST preserve ALL Agent 4 outputs EXACTLY, including:
- Merchant type
- PD score (exact value)
- Risk review type
- Documents required
- Any compliance artifact indicators

✅ REQUIRED HANDLING
A. Narrative Explanation
Explain what the PD score means, what the merchant type implies, and what the review type indicates.

B. 📄 Required Documentation (MANDATORY SECTION)
You MUST include a clearly visible section titled: "📄 Required Documentation (To Be Collected from Merchant)"
Include: PD value check (4.00%), DDD documents, Questionnaire, Playbook link.
🚫 Do NOT: Remove items, Rename items, or Convert into paragraph-only format.

C. Sales Interpretation
Explain why each document is needed, that these are mandatory for onboarding progression, and that delays will block onboarding.

⚖️ RISK HANDLING GUIDELINES
Use this priority when interpreting signals: 1. Adverse Media (highest), 2. Regulatory / Compliance exposure, 3. Risk scoring, 4. Financial profile, 5. Sentiment (lowest).
Explain conflicts clearly in simple business terms.

📄 REQUIRED OUTPUT STRUCTURE
🧾 MERCHANT ONBOARDING REPORT (SALES ENABLEMENT)

1. Executive Overview
- High-level summary, Recommendation-style conclusion, Key strengths and considerations, Emphasize dependency on documentation.

2. Merchant Snapshot
- Merchant name, Industry, Website, Merchant type, MCC (if available), Business context.

3. Data Validation Status
- Data quality score, Completeness, Issues (if any), 👉 Include Sales implication.

4. Risk & Compliance Overview (Simplified)
- Risk level & score, Review requirements, Compliance expectations, 👉 Translate into Sales-friendly interpretation.

5. Financial & Documentation Requirements (CRITICAL)
- Narrative Explanation: PD score, Merchant type, Review type.
- 📄 Required Documentation: PD value check (4.00%), DDD documents, Questionnaire, Playbook link.
- 📌 Sales Guidance: Explain what each is, why it’s required, impact if missing.
- ⚠️ Important for Sales: Clearly state these are mandatory, missing documents block onboarding, these are standard requirements.

6. Adverse Media Considerations (Sales Awareness)
- Summarize findings, Avoid alarmist tone, Emphasize internal awareness, 👉 Include Sales implication.

7. Customer Sentiment Overview
- Sentiment score, Rating, Interpretation.

8. Overall Risk Positioning (Sales Framing)
- Strengths, Considerations, Balanced view, 👉 Frame as: “Viable opportunity with conditions”.

9. Recommended Next Steps (FOR SALES TEAM)
- 🚀 Immediate Actions: What to collect, What to send.
- 📊 Coordination: Internal alignment.
- ⏱️ Follow-Up: Tracking expectations.

10. Summary for Sales
- 3–5 line takeaway emphasizing opportunity viability and documentation as key blocker.

⚠️ STRICT RULES
- Do NOT output JSON
- Do NOT skip sections
- Do NOT remove Agent 4 details
- Do NOT state a final decision
- Do NOT over-compress document requirements

✅ OUTPUT
Return ONLY the final structured report.
"""

def generate_onboarding_summary(orchestration_decision: dict,
                                  evidence_metadata: dict) -> dict:
    # strip enum reprs and trace noise
    compact_payload = _build_compact_payload(orchestration_decision, evidence_metadata)

    user_message = (
        "Generate the merchant onboarding report based on the following "
        "structured data. Follow all formatting and sales-enablement instructions strictly:\n\n"
        f"{json.dumps(compact_payload, indent=2, default=str)}"
    )

    # Call the LLM
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        summary_text = response.choices[0].message.content.strip()
        model_used = response.model
    except Exception as e:
        summary_text = f"[Summary generation failed: {e}]"
        model_used = "error"

    return {
        "summary_text": summary_text,
        "model_used": model_used,
        "system_instructions": SYSTEM_PROMPT,
        "input_payload": compact_payload,
    }

def _build_compact_payload(orchestration_decision: dict,
                            evidence_metadata: dict) -> dict:
    # Final decision (convert enum to plain string if present)
    decision = orchestration_decision.get("final_decision", "")
    if hasattr(decision, "value"):
        decision = decision.value
    decision = str(decision)

    # Pull adverse media findings (top 5 only)
    adverse = evidence_metadata.get("adverse_media", {}) or {}
    findings = adverse.get("findings", []) or []
    top_findings = []
    for f in findings[:5]:
        top_findings.append({
            "title": f.get("title", "")[:200],
            "category": f.get("category", ""),
            "severity": f.get("severity", ""),
            "summary": f.get("summary", "")[:300],
        })

    # Pull document requirements
    docs = evidence_metadata.get("document_requirements", {}) or {}

    # Pull risk assessment (just the structured fields, not raw_output)
    risk = evidence_metadata.get("risk_assessment", {}) or {}

    # Pull sentiment summary
    sentiment = evidence_metadata.get("sentiment", {}) or {}

    return {
        "merchant_name": orchestration_decision.get("merchant_name", ""),
        "final_decision": decision,
        "decision_rationale": orchestration_decision.get("decision_rationale", ""),
        "escalation_reasons": orchestration_decision.get("escalation_reasons", []),

        "risk_assessment": {
            "risk_level": risk.get("risk_level", "UNKNOWN"),
            "risk_score": risk.get("risk_score"),
            "compliance_review_required": risk.get("compliance_review_required", False),
            "required_questionnaires": risk.get("required_questionnaires", []),
            "applicable_playbooks": risk.get("applicable_playbooks", []),
        },

        "adverse_media": {
            "severity": adverse.get("overall_risk_severity", "NONE"),
            "score": adverse.get("overall_risk_score", 0),
            "data_sufficiency": adverse.get("data_sufficiency", "UNKNOWN"),
            "top_findings": top_findings,
        },

        "sentiment": {
            "overall_score": sentiment.get("overall_sentiment_score", 0),
            "satisfaction_rating": sentiment.get("customer_satisfaction_rating", 0),
            "review_count": sentiment.get("review_count", 0),
        },

        "documents": {
            "required_documents": docs.get("required_documents", []),
            "deadline_days": docs.get("document_submission_deadline_days", 7),
        },
    }
