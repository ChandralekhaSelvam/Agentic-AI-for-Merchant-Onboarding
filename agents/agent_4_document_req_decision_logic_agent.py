"""
================================================================================
AGENT 4 — DOCUMENT REQUIREMENTS & DECISION LOGIC AGENT
================================================================================
Purpose: Determine exact document requirements based on merchant type and
         risk profile.

Inputs (from Agents 2 and 3):
    - Validated merchant profile (name, url, industry)
    - Risk review flag (yes/no)
    - Compliance review flag (yes/no)

Process:
    1. If risk review = NO and compliance review = NO → return early
    2. If risk review = YES:
        a. Look up merchant in SQLite DB (merchant.db)
        b. If NEW merchant → full document set
        c. If EXISTING merchant → classify by PD score:
             pd <= 0.02       → Light Review
             0.02 < pd <= 0.05 → Medium Review
             pd > 0.05        → Full Review
    3. If compliance review = YES:
        a. Read compliance_requirements.md
        b. Use GPT-4o-mini to check if merchant's industry is high-risk
        c. Return artifacts required
    4. Combine risk + compliance outputs into a markdown report (via LLM)

Outputs:
    - merchant_type: New | Existing
    - risk_review: Light | Medium | Full Review
    - documents_required: List[str]
    - compliance_documentation: Dict with artifacts
    - full_report_markdown: Human-readable report

Run:  pip install langchain-openai langchain-core pandas pdfplumber
      export OPENAI_API_KEY="sk-..."
      python agent_4_document_requirements.py
================================================================================
"""

import os
import json
import sqlite3
import datetime
from typing import TypedDict, Dict, Any, Optional, List

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH = "merchant.db"
COMPLIANCE_MD_PATH = "data/compliance_requirements.md"
MODEL_NAME = "gpt-4o-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY environment variable")


# =============================================================================
# STATE SCHEMA
# =============================================================================

class AgentState(TypedDict, total=False):
    """Input state for Agent 4 — comes from Agents 2 and 3."""
    merchant_name: str
    merchant_url: str
    industry: str
    risk_review: str           # "yes" | "no"
    compliance_review: str     # "yes" | "no"
    playbook: str
    playbook_link: str
    questionnaire: str
    questionnaire_link: str
    full_report_markdown: str


# =============================================================================
# DATABASE SETUP — Seed sample merchants
# =============================================================================

def initialize_database(db_path: str = DB_PATH) -> None:
    """
    Create the merchants table and seed it with sample data.

    In production, this DB would be populated from BigQuery with real merchant
    history and PD scores. For demo purposes we seed a few well-known merchants.
    """
    # Start fresh
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS merchants")
    cursor.execute("""
        CREATE TABLE merchants (
            merchant_name TEXT,
            merchant_url  TEXT,
            industry      TEXT,
            country       TEXT,
            pd_score      REAL
        )
    """)

    seed_data = [
        ("Amazon",              "amazon.com",            "E-commerce",      "USA",         0.010),
        ("Walmart",             "walmart.com",           "Retail",          "USA",         0.015),
        ("Apple",               "apple.com",             "Technology",      "USA",         0.008),
        ("Microsoft",           "microsoft.com",         "Technology",      "USA",         0.012),
        ("Alphabet",            "abc.xyz",               "Technology",      "USA",         0.011),
        ("Meta",                "meta.com",              "Technology",      "USA",         0.020),
        ("Netflix",             "netflix.com",           "Entertainment",   "USA",         0.025),
        ("Tesla",               "tesla.com",             "Automotive",      "USA",         0.030),
        ("Reliance Industries", "ril.com",               "Conglomerate",    "India",       0.022),
        ("Tata Group",          "tata.com",              "Conglomerate",    "India",       0.018),
        ("Infosys",             "infosys.com",           "IT Services",     "India",       0.017),
        ("Accenture",           "accenture.com",         "Consulting",      "Ireland",     0.014),
        ("Stripe",              "stripe.com",            "Fintech",         "USA",         0.070),
        ("PayPal",              "paypal.com",            "Fintech",         "USA",         0.040),
        ("Shopify",             "shopify.com",           "E-commerce",      "Canada",      0.050),
        ("Adobe",               "adobe.com",             "Software",        "USA",         0.016),
        ("Salesforce",          "salesforce.com",        "SaaS",            "USA",         0.019),
        ("Oracle",              "oracle.com",            "Technology",      "USA",         0.021),
        ("IBM",                 "ibm.com",               "Technology",      "USA",         0.023),
        ("Intel",               "intel.com",             "Semiconductors",  "USA",         0.020),
        ("Cisco",               "cisco.com",             "Networking",      "USA",         0.018),
        ("Samsung",             "samsung.com",           "Electronics",     "South Korea", 0.013),
        ("Sony",                "sony.com",              "Electronics",     "Japan",       0.022),
        ("Uber",                "uber.com",              "Mobility",        "USA",         0.055),
        ("Airbnb",              "airbnb.com",            "Hospitality",     "USA",         0.060),
        ("Booking Holdings",    "bookingholdings.com",   "Travel",          "USA",         0.035),
        ("Nike",                "nike.com",              "Apparel",         "USA",         0.017),
        ("Adidas",              "adidas.com",            "Apparel",         "Germany",     0.019),
        ("Coca-Cola",           "coca-cola.com",         "Beverages",       "USA",         0.012),
        ("PepsiCo",             "pepsico.com",           "Beverages",       "USA",         0.013),
    ]

    cursor.executemany(
        "INSERT INTO merchants (merchant_name, merchant_url, industry, country, pd_score) "
        "VALUES (?, ?, ?, ?, ?)",
        seed_data,
    )
    conn.commit()
    conn.close()
    print(f"📂 Initialized {db_path} with {len(seed_data)} seed merchants")


def fetch_merchant(name: str, url: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Look up a merchant by name or URL (case-insensitive).
    Returns an empty DataFrame if not found.
    """
    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT * FROM merchants
            WHERE LOWER(merchant_name) = LOWER(?)
               OR LOWER(merchant_url)  = LOWER(?)
        """
        df = pd.read_sql_query(query, conn, params=(name, url))
    finally:
        conn.close()
    return df


# =============================================================================
# COMPLIANCE REQUIREMENTS DOCUMENT
# =============================================================================

COMPLIANCE_MD_CONTENT = """# High Risk Vertical Compliance Requirements

This document lists high-risk industries along with the typical compliance
artifacts required for onboarding, verification, and regulatory adherence.

| High Risk Industry              | Artifacts Required                                                                 |
|--------------------------------|------------------------------------------------------------------------------------|
| Financial Services (Lending)    | Business Verification, Signed Application Form, KYC (Directors), Loan Agreements, Credit Policy Documents |
| Cryptocurrency / Web3          | Business Verification, KYC (All Founders), AML Policy, Wallet Address Proof, Transaction Monitoring Setup |
| Gambling / Betting             | Gaming License, Business Registration Proof, KYC, AML Policy, Geo-restriction Controls |
| Adult Entertainment            | Business Verification, Age Compliance Policy, Content Compliance Declaration, KYC |
| Pharmaceuticals / Online Pharmacy | Drug License, Business Registration, KYC, Prescription Handling Policy, Compliance Certificates |
| Insurance Aggregators          | IRDAI License (or equivalent), Business Verification, KYC, Signed Agreements, Policy Documents |
| Forex / Trading Platforms      | Regulatory License, KYC (Directors), AML Policy, Risk Disclosure Documents |
| Precious Metals / Jewelry      | Business Registration, GST Certificate, KYC, Supplier Invoices, Inventory Records |
| Travel & Ticketing             | Business Verification, IATA License (if applicable), KYC, Refund Policy, Vendor Agreements |
| Multi-level Marketing (MLM)    | Business Registration, KYC, Compensation Plan Details, Legal Compliance Declaration |
| Crowdfunding Platforms         | Business Verification, KYC (Founders), Escrow Agreements, AML Policy, Investor Terms |
| Gaming (Real Money)            | Gaming License, RNG Certification, KYC, AML Policy, User Protection Measures |
| Alcohol / Tobacco Sales        | Trade License, Business Registration, Age Verification Policy, KYC |
| Digital Lending Apps           | NBFC Partnership Proof, KYC, Loan Agreements, Data Privacy Policy, Credit Model Documentation |
| Data Brokerage / Lead Gen      | Data Source Declaration, Consent Collection Proof, Privacy Policy, KYC, Contracts |

---

## Notes
- **KYC**: Know Your Customer documentation for directors/owners (ID proof, address proof, etc.)
- **Business Verification**: Includes incorporation certificate, tax registration, and operational proof
- **AML Policy**: Anti-Money Laundering compliance documentation
- Additional artifacts may be required based on jurisdiction and regulatory changes
"""


def ensure_compliance_md(path: str = COMPLIANCE_MD_PATH) -> None:
    """Write compliance_requirements.md if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(COMPLIANCE_MD_CONTENT)
        print(f"📄 Created {path}")


# =============================================================================
# LLM INITIALIZATION
# =============================================================================

llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)


# =============================================================================
# RISK AGENT — Core decision logic
# =============================================================================

def risk_agent(state: AgentState) -> Dict[str, Any]:
    """
    Determine document requirements based on risk review flag and merchant status.

    Logic:
        - risk = NO, compliance = YES  → no docs from risk side
        - risk = NO, compliance = NO   → nothing required
        - risk = YES:
            * New merchant → full document set
            * Existing merchant → tier by PD score:
                pd <= 0.02       → Light Review (minimal docs)
                0.02 < pd <= 0.05 → Medium Review (DDD docs added)
                pd > 0.05        → Full Review (all financials)
    """
    risk_flag = state.get("risk_review", "").lower()
    compliance_flag = state.get("compliance_review", "").lower()

    # Case 1: Risk not required, compliance required
    if risk_flag == "no" and compliance_flag == "yes":
        return {
            "risk_review": "Not Required",
            "compliance_review": "Required",
            "message": "Risk review not required, but compliance review is required",
        }

    # Case 2: Neither required
    if risk_flag == "no" and compliance_flag == "no":
        return {
            "risk_review": "Not Required",
            "compliance_review": "Not Required",
            "message": "Neither risk review nor compliance review is required",
        }

    # Case 3: Risk required — look up merchant
    if risk_flag == "yes":
        name = state["merchant_name"]
        url = state.get("merchant_url", "")

        result = fetch_merchant(name, url)

        # New merchant — no history in DB
        if result.empty:
            return {
                "merchant_type": "New Merchant",
                "risk_review": "Full Review",
                "documents_required": [
                    "Full financials",
                    "Bank statement",
                    "Interim financials",
                    "Processing statements",
                ],
            }

        # Existing merchant — tier by PD score
        pd_value = float(result.iloc[0]["pd_score"])

        if pd_value <= 0.02:
            return {
                "merchant_type": "Existing Merchant",
                "risk_review": "Light Review",
                "pd_score": pd_value,
                "documents_required": [
                    f"PD value check ({pd_value*100:.2f}%)",
                    "Questionnaire",
                    "Playbook link",
                ],
            }

        if pd_value <= 0.05:
            return {
                "merchant_type": "Existing Merchant",
                "risk_review": "Medium Review",
                "pd_score": pd_value,
                "documents_required": [
                    f"PD value check ({pd_value*100:.2f}%)",
                    "DDD documents",
                    "Questionnaire",
                    "Playbook link",
                ],
            }

        return {
            "merchant_type": "Existing Merchant",
            "risk_review": "Full Review",
            "pd_score": pd_value,
            "documents_required": [
                "Full financials",
                "Bank statement",
                "Interim financials",
                "Processing statements",
            ],
        }

    # Fallback for unexpected risk_flag values
    return {
        "risk_review": "Unknown",
        "message": f"Unrecognized risk_review value: {state.get('risk_review')}",
    }


# =============================================================================
# COMPLIANCE AGENT — LLM-backed high-risk industry check
# =============================================================================

def compliance_agent(state: AgentState) -> Dict[str, Any]:
    """
    Use GPT-4o-mini to check if the merchant's industry is flagged as high-risk
    in compliance_requirements.md, and return the required artifacts if so.
    """
    compliance_flag = state.get("compliance_review", "").lower()

    if compliance_flag == "no":
        return {"compliance_documentation": "Compliance review not required."}

    merchant_industry = state.get("industry", "")

    # Read compliance requirements document
    try:
        with open(COMPLIANCE_MD_PATH, "r") as f:
            markdown_content = f.read()
    except FileNotFoundError:
        return {
            "compliance_documentation": {
                "merchant_industry": merchant_industry,
                "high_risk_match": False,
                "message": f"Error: {COMPLIANCE_MD_PATH} not found.",
            }
        }

    # Build LLM prompt
    prompt = PromptTemplate.from_template(
        """Given the following compliance requirements document:

```markdown
{document}
```

And a merchant with the industry '{industry}'.

Determine if this industry is considered 'high risk' according to the document.
If it is, list all the 'Artifacts Required'. If not, state that it is not high risk.

Provide your answer in a JSON format with two keys:
- `is_high_risk` (boolean, true if high risk, false otherwise)
- `artifacts` (list of strings if high risk, or a string message if not)

Return only the JSON object, no other text."""
    )

    formatted_prompt = prompt.format(
        document=markdown_content,
        industry=merchant_industry,
    )

    # Call LLM with error handling
    response_content = ""
    json_string = ""
    try:
        llm_response = llm.invoke(formatted_prompt)
        response_content = llm_response.content

        # Extract JSON from possible markdown fences
        start_idx = response_content.find("{")
        end_idx = response_content.rfind("}")

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return {
                "compliance_documentation": {
                    "merchant_industry": merchant_industry,
                    "high_risk_match": False,
                    "message": f"Could not extract JSON from LLM response: {response_content[:200]}",
                }
            }

        json_string = response_content[start_idx:end_idx + 1]
        llm_output = json.loads(json_string)

    except json.JSONDecodeError as e:
        return {
            "compliance_documentation": {
                "merchant_industry": merchant_industry,
                "high_risk_match": False,
                "message": f"Error parsing LLM JSON: {e}. Extracted: {json_string[:200]}",
            }
        }
    except Exception as e:
        return {
            "compliance_documentation": {
                "merchant_industry": merchant_industry,
                "high_risk_match": False,
                "message": f"Error calling LLM: {e}",
            }
        }

    # Process LLM output
    is_high_risk = bool(llm_output.get("is_high_risk", False))
    artifacts = llm_output.get("artifacts", "Could not determine artifacts.")

    if is_high_risk:
        return {
            "compliance_documentation": {
                "merchant_industry": merchant_industry,
                "high_risk_match": True,
                "artifacts_required": artifacts if isinstance(artifacts, list) else [str(artifacts)],
            }
        }

    return {
        "compliance_documentation": {
            "merchant_industry": merchant_industry,
            "high_risk_match": False,
            "message": artifacts if isinstance(artifacts, str) else str(artifacts),
        }
    }


# =============================================================================
# REPORT GENERATOR — Combine risk + compliance into a markdown report
# =============================================================================

def generate_assessment_report_markdown(
    state: AgentState,
    risk_output: Dict[str, Any],
    compliance_output_doc: Dict[str, Any],
    llm_model=llm,
) -> str:
    """
    Use the LLM to produce a polished markdown report combining risk and
    compliance outputs. Output is pure markdown, ready for PDF conversion.
    """
    merchant_name = state.get("merchant_name", "Unknown")
    merchant_url = state.get("merchant_url", "N/A")
    industry = state.get("industry", "N/A")

    # Build risk bullets
    risk_bullets: List[str] = []
    risk_bullets.append(f"- **Merchant Type:** {risk_output.get('merchant_type', 'N/A')}")
    risk_bullets.append(f"- **Review Type:** {risk_output.get('risk_review', 'N/A')}")
    if "pd_score" in risk_output:
        risk_bullets.append(
            f"- **Probability of Default (PD) Score:** {risk_output['pd_score']:.4f} "
            f"(lower is better)"
        )
    docs_required = risk_output.get("documents_required", [])
    if docs_required:
        risk_bullets.append("### Documents Required for Risk Assessment:")
        for doc in docs_required:
            risk_bullets.append(f"  - {doc}")
    else:
        risk_bullets.append("  - No specific risk documents required.")
    risk_agent_bullets = "\n".join(risk_bullets)

    # Build compliance bullets
    comp_bullets: List[str] = []
    if isinstance(compliance_output_doc, dict):
        comp_bullets.append(
            f"- **Merchant Industry:** {compliance_output_doc.get('merchant_industry', 'N/A')}"
        )
        is_high_risk = compliance_output_doc.get("high_risk_match", False)
        comp_bullets.append(
            f"- **High Risk Industry Match:** {'Yes' if is_high_risk else 'No'}"
        )

        if is_high_risk:
            artifacts = compliance_output_doc.get("artifacts_required", [])
            if artifacts:
                comp_bullets.append("### Compliance Artifacts Required:")
                if isinstance(artifacts, list):
                    for artifact in artifacts:
                        comp_bullets.append(f"  - {artifact}")
                else:
                    comp_bullets.append(f"  - {artifacts}")
            else:
                comp_bullets.append("  - No artifacts listed despite high-risk classification.")
        else:
            comp_bullets.append(
                f"- **Compliance Message:** {compliance_output_doc.get('message', 'N/A')}"
            )
    else:
        comp_bullets.append(f"- {compliance_output_doc}")
    compliance_agent_details = "\n".join(comp_bullets)

    current_date = datetime.date.today().strftime("%Y-%m-%d")

    prompt_template = PromptTemplate.from_template(
        """Based on the merchant's assessment data below, generate a descriptive
markdown report. Return only markdown, no conversational text, no code fences.

Merchant Information:
- Name: {merchant_name}
- URL: {merchant_url}
- Industry: {industry}

Risk Agent Output (raw):
```json
{risk_output_str}
```

Compliance Agent Output (raw):
```json
{compliance_output_str}
```

Use this structure exactly:

# Merchant Assessment Report for {merchant_name}

## Merchant Overview

- **Merchant Name:** {merchant_name}
- **Merchant URL:** {merchant_url}
- **Primary Industry:** {industry}

## Risk Documentation Analysis

{risk_agent_bullets}

## Compliance Documentation Analysis

{compliance_agent_details}

---

*This report is generated based on automated agent assessments as of {current_date}.*
"""
    )

    formatted_prompt = prompt_template.format(
        merchant_name=merchant_name,
        merchant_url=merchant_url,
        industry=industry,
        risk_output_str=json.dumps(risk_output, default=str, indent=2),
        compliance_output_str=json.dumps(compliance_output_doc, default=str, indent=2),
        risk_agent_bullets=risk_agent_bullets,
        compliance_agent_details=compliance_agent_details,
        current_date=current_date,
    )

    print("  Generating descriptive report markdown with LLM...")
    llm_response = llm_model.invoke(formatted_prompt)
    return llm_response.content


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_document_requirements(state: AgentState) -> Dict[str, Any]:
    """
    Main Agent 4 entry point. Runs the full pipeline for a single merchant.

    Args:
        state: AgentState dict with merchant info and risk/compliance flags.

    Returns:
        Dict containing risk output, compliance output, and full markdown report.
    """
    merchant_name = state.get("merchant_name", "Unknown")

    print(f"\n{'='*70}")
    print(f"  AGENT 4 — DOCUMENT REQUIREMENTS & DECISION LOGIC")
    print(f"  Merchant: {merchant_name}")
    print(f"{'='*70}\n")

    # Make sure data files exist
    if not os.path.exists(DB_PATH):
        initialize_database(DB_PATH)
    ensure_compliance_md(COMPLIANCE_MD_PATH)

    # Step 1: Risk assessment
    print("  [1/3] Running risk agent...")
    risk_output = risk_agent(state)
    print(f"       → {risk_output.get('merchant_type', risk_output.get('risk_review', 'N/A'))}")

    # Step 2: Compliance assessment
    print("  [2/3] Running compliance agent...")
    compliance_output = compliance_agent(state)
    compliance_doc = compliance_output.get("compliance_documentation", {})
    if isinstance(compliance_doc, dict):
        high_risk = compliance_doc.get("high_risk_match", False)
        print(f"       → High-risk industry: {'Yes' if high_risk else 'No'}")
    else:
        print(f"       → {compliance_doc}")

    # Step 3: Generate combined report
    print("  [3/3] Generating combined report...")
    report_markdown = generate_assessment_report_markdown(
        state=state,
        risk_output=risk_output,
        compliance_output_doc=compliance_doc if isinstance(compliance_doc, dict) else {},
        llm_model=llm,
    )

    result = {
        "merchant_name": merchant_name,
        "risk_output": risk_output,
        "compliance_output": compliance_output,
        "full_report_markdown": report_markdown,
        "documents_required": risk_output.get("documents_required", []),
        "is_existing_merchant": risk_output.get("merchant_type") == "Existing Merchant",
        "pd_score": risk_output.get("pd_score"),
    }

    print(f"\n  ✓ Agent 4 complete for {merchant_name}\n")
    return result


# =============================================================================
# MAIN — Standalone test
# =============================================================================

if __name__ == "__main__":

    # Initialize DB and compliance doc on first run
    initialize_database()
    ensure_compliance_md()

    # Show what's in the DB
    conn = sqlite3.connect(DB_PATH)
    df_merchants = pd.read_sql_query("SELECT * FROM merchants LIMIT 5", conn)
    conn.close()
    print("\nSample merchants in DB:")
    print(df_merchants)

    # Test case 1: Existing merchant, low risk
    print("\n" + "─" * 70)
    print("TEST 1: Amazon (existing merchant, low PD)")
    print("─" * 70)

    state_amazon: AgentState = {
        "merchant_name": "Amazon",
        "merchant_url": "amazon.com",
        "risk_review": "yes",
        "compliance_review": "yes",
        "industry": "E-commerce",
        "playbook": "yes",
        "playbook_link": "http://playbook.com",
        "questionnaire": "yes",
        "questionnaire_link": "http://questionnaire.com",
    }

    result_amazon = run_document_requirements(state_amazon)
    print("\n📋 RISK OUTPUT:")
    print(json.dumps(result_amazon["risk_output"], indent=2, default=str))
    print("\n📋 COMPLIANCE OUTPUT:")
    print(json.dumps(result_amazon["compliance_output"], indent=2, default=str))

    # Save the markdown report
    report_file = f"{state_amazon['merchant_name'].replace(' ', '_')}_assessment_report.md"
    with open(report_file, "w") as f:
        f.write(result_amazon["full_report_markdown"])
    print(f"\n📁 Report saved: {report_file}")

    # Test case 2: High-risk industry (Gambling)
    print("\n" + "─" * 70)
    print("TEST 2: New merchant in Gambling (high-risk industry)")
    print("─" * 70)

    state_typed: AgentState = {
        "merchant_name": "BetKing Online",
        "merchant_url": "betking.example",
        "risk_review": "yes",
        "compliance_review": "yes",
        "industry": "Gambling",
        "playbook": "yes",
        "playbook_link": "http://playbook.com",
        "questionnaire": "yes",
        "questionnaire_link": "http://questionnaire.com",
    }

    result_gambling = run_document_requirements(state_typed)
    print("\n📋 RISK OUTPUT:")
    print(json.dumps(result_gambling["risk_output"], indent=2, default=str))
    print("\n📋 COMPLIANCE OUTPUT:")
    print(json.dumps(result_gambling["compliance_output"], indent=2, default=str))