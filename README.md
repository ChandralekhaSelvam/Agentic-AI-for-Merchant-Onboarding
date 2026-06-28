# Agentic AI for Merchant Onboarding

> **Multi-Agent Intelligent Onboarding Pipeline**

| | |
|---|---|
| **Document Version** | v1.0.0 |
| **Classification** | Internal — Engineering & Compliance |
| **Platform** | Python 3.11+ / Flask / LangChain / LangGraph |
| **AI Models** | GPT-4o-mini (OpenAI) + HuggingFace Transformers |
| **Architecture** | Multi-Agent Sequential + Parallel Pipeline |
| **Interfaces** | REST API / Server-Sent Events / Web UI |

---

> ### 🎓 Academic Context
>
> This project is submitted as the **capstone deliverable** for the **PG Certificate Program in Generative AI & Agentic AI** at **IIT Roorkee**. It demonstrates the end-to-end design, implementation, and orchestration of a production-grade multi-agent AI system — bringing together core concepts from the program (LLM reasoning, tool use, multi-agent orchestration, and responsible AI) into a single enterprise-grade pipeline applied to real-world merchant onboarding and risk decisioning.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Agent Reference](#3-agent-reference)
4. [Pipeline Orchestrator](#4-pipeline-orchestrator)
5. [Web UI & API Server](#5-web-ui--api-server)
6. [Installation & Setup](#6-installation--setup)
7. [Running the System](#7-running-the-system)
8. [Dependencies](#8-dependencies)
9. [Configuration & Environment Variables](#9-configuration--environment-variables)
10. [Troubleshooting](#10-troubleshooting)
11. [Security & Compliance Considerations](#11-security--compliance-considerations)

---

## 1. Executive Summary

The **Agentic AI for Merchant Onboarding** platform is an enterprise-grade, multi-agent intelligent system engineered to automate, standardize, and accelerate the merchant onboarding and risk assessment process for financial institutions and payment processors.

The platform orchestrates eight specialized AI agents in a configurable pipeline — combining rule-based policy engines, large language model (LLM) reasoning, machine learning sentiment analysis, adverse media screening, and structured compliance workflows — to produce a deterministic, auditable onboarding decision for every merchant.

### 1.1 Business Problem

Traditional merchant onboarding is slow, inconsistent, and resource-intensive. Analysts must manually cross-reference risk policies, screen for negative news, assess financial documents, and produce compliance reports — a process that can take days and is prone to human error and inconsistent policy application.

### 1.2 Solution Overview

**Platform Capabilities:**

- Automated data ingestion and structured validation of merchant profiles
- Policy-driven risk & compliance scoring using Excel-backed rule sets
- Probability of Default (PD) scoring for document requirement tiering
- Real-time adverse media screening across multiple public data sources
- ML-powered customer sentiment analysis using transformer models
- LLM-orchestrated final decision: `APPROVE` / `MANUAL_REVIEW` / `REJECT`
- Narrative onboarding report generation suitable for Sales and Risk teams
- Real-time streaming web UI with per-agent visibility and JSON inspection

### 1.3 Key Outcomes

| Metric | Manual Process | This Platform |
|---|---|---|
| **Assessment Time** | 2–5 Business Days | < 5 Minutes |
| **Policy Consistency** | Analyst-dependent | 100% Rule-enforced |
| **Adverse Media Coverage** | Limited / Sampled | Multi-source, automated |
| **Audit Trail** | Manual notes | Full structured JSON |
| **Parallel Processing** | Not possible | 3 agents simultaneously |

---

## 2. System Architecture

### 2.1 High-Level Architecture

The system is structured as a sequential-and-parallel multi-agent pipeline. Each agent is a discrete, independently testable module with clearly defined inputs, outputs, and failure modes. Agents communicate through structured Python dictionaries that enforce type safety and schema validation.

### 2.2 Pipeline Execution Flow

```
Stage 1:  Agent 1 (Data Ingestion)        → SQLite persistence
Stage 2:  Agent 2 (Validation)            → Profile normalization & scoring
Stage 3:  Agents 3, 5, 6 run in PARALLEL  (ThreadPoolExecutor)
          ├── Agent 3: Risk & Compliance Assessment
          ├── Agent 5: Negative News / Adverse Media Detection
          └── Agent 6: Customer Sentiment Analysis
Stage 4:  Agent 4 (Document Requirements) → runs after Agent 3
Stage 5:  Agent 7 (Orchestration)         → LangGraph state machine
Stage 6:  Agent 8 (Summarization)         → Final narrative report
```

### 2.3 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Runtime | Python 3.11+ | Core agent execution environment |
| LLM Provider | OpenAI GPT-4o-mini | Reasoning, classification, report generation |
| Agent Framework | LangChain 0.2.14 | Tool orchestration and agent prompting |
| State Machine | LangGraph 0.2.16 | Agent 7 orchestration graph |
| Sentiment ML | HuggingFace Transformers | Transformer-based sentiment classification |
| ML Utilities | scikit-learn, TextBlob | TF-IDF, clustering, sentiment scoring |
| Deep Learning | PyTorch 2.2.2, TensorFlow | Transformer model inference |
| Policy Storage | Excel (openpyxl/pandas) | Risk rules, MCC mappings, questionnaires |
| Merchant DB | SQLite (via sqlite3) | Merchant history and PD scores |
| Web Server | Flask 3.0+ / Flask-CORS | REST API and SSE streaming |
| UI | Vanilla HTML/CSS/JS | Real-time pipeline monitoring |
| Parallelism | ThreadPoolExecutor | Concurrent agent execution |
| Data Parsing | BeautifulSoup4, requests | Web scraping for news and reviews |
| Config | python-dotenv | Environment variable management |

---

## 3. Agent Reference

Each agent in the pipeline is a self-contained Python module. Below is a comprehensive specification for every agent, including its purpose, inputs, outputs, tools used, and configuration details.

### 3.1 Agent 1 — Data Ingestion Agent

> **File:** `agent_1_data_ingestion.py` | **Class:** `DataIngestionAgent`

Responsible for accepting merchant data, normalizing it, and persisting it to a SQLite database. Supports both single-merchant direct insertion and bulk ingestion from Excel (XLSX) files. Serves as the authoritative data store for all downstream agents.

#### Inputs

- **Single merchant:** `name` (required), `address`, `website`, `industry`, `mcc`, `additional_data`
- **Bulk:** Excel file (`.xlsx`) with columns: `name`, `address`, `website`, `industry`, `mcc`, `additional_data`

#### Outputs

- Persisted merchant record with auto-incremented `merchant_id`
- Normalized data: URLs prefixed with `https://`, text title-cased, MCC zero-padded

#### Database Schema

```sql
CREATE TABLE merchants (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    merchant_name   TEXT,
    address         TEXT,
    website         TEXT,
    industry        TEXT,
    mcc             TEXT,
    additional_data TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Key Methods

| Method | Description |
|---|---|
| `insert_merchant()` | Insert a single merchant record with normalization |
| `ingest_from_xlsx(file_path)` | Bulk load merchants from an Excel file |
| `fetch_all()` | Retrieve all stored merchant records |
| `clear_all_data()` | Truncate the merchants table (use with caution) |

---

### 3.2 Agent 2 — Validation Agent

> **File:** `agent_2_validation_agent.py` | **Class:** `ValidationAgent`

Validates the merchant profile for completeness, field format correctness, taxonomy alignment, and risk classification. Produces a data quality score (0–100) and a validation status (`valid` / `partial` / `invalid`). This is the only agent whose failure can cause early pipeline termination.

#### Validation Rules

- **Required fields check:** `merchant_name`, `address`, `website`, `industry`, `mcc`
- **Website format validation:** must match `https?://[\w-]+(\.[\w-]+)+` pattern
- **MCC code validation:** must be a 4-digit code present in the policy MCC mapping
- **Industry taxonomy normalization:** aliases resolved to canonical industry names
- **High-risk flag detection:** MCC or industry scoring `>= 80` triggers flag

#### Scoring Logic

```python
base_score      = 100
missing_penalty = len(missing_fields) * 15
issue_penalty   = len(issues) * 10
risk_penalty    = 25 per HIGH-risk flag + 10 per MEDIUM-risk flag

final_score = max(0, base_score - missing_penalty - issue_penalty - risk_penalty)

# Status:
#   score >= 85  → 'valid'
#   score >= 60  → 'partial'   (pipeline continues with warnings)
#   score <  60  → 'invalid'   (pipeline terminates early → REJECT)
```

#### Output Schema

```json
{
  "validated_profile": {
    "...normalized fields": "...",
    "data_quality_score": 0,
    "status": "valid | partial | invalid"
  },
  "data_quality_report": {
    "score": 0,
    "status": "string",
    "missing_fields": [],
    "issues": [
      { "field": "...", "issue": "...", "suggestion": "...", "code": "..." }
    ],
    "flagged_items": [
      { "field": "...", "risk_level": "...", "risk_score": 0, "reason": "...", "code": "..." }
    ]
  },
  "flagged_items": []
}
```

#### Policy Data

If a policy Excel file is provided (`policy_xlsx` parameter), the agent loads industry taxonomy, MCC mappings, risk flags, and TPB rules from it. Without the file, hardcoded fallback defaults are used (covering Retail, Gambling, Crypto, Cannabis, Travel, Healthcare, Food Services, FinTech).

---

### 3.3 Agent 3 — Risk & Compliance Assessment Agent

> **File:** `agent_3_risk_compliance.py` | **Framework:** LangChain ReAct Agent with 6 custom tools

Performs a comprehensive risk and compliance assessment using a LangChain-powered agent executor. The agent loads all policy sheets from an Excel file and applies six sequential tools to classify the merchant's risk profile, evaluate thresholds, determine questionnaire requirements, map compliance playbooks, and produce a final assessment.

#### Policy Data Sources (Excel Sheets)

- **`Industry_Risk_Mapping`** — Industry → MCC, Risk Level, Risk Score, High Risk Flag
- **`TPB_Thresholds`** — Third Party Business percentage rules by risk tier
- **`Questionnaire_Templates`** — Q-001 through Q-010 with applicability rules
- **`Compliance_Playbooks`** — PB-001 through PB-013 with regulatory frameworks
- **`Risk_Decision_Logic`** — RULE-001 through RULE-012 decision conditions

#### Agent Tool Sequence (Mandatory Order)

| Step | Tool | Purpose |
|:---:|---|---|
| 1 | `lookup_industry_risk` | Classify merchant industry risk: level, score, high-risk flag, reason |
| 2 | `evaluate_tpb_threshold` | Check if TPB % triggers additional review requirements |
| 3 | `get_required_questionnaires` | Determine Q-001–Q-010 based on risk, industry, volume, chargeback rate |
| 4 | `get_applicable_playbooks` | Map merchant to applicable PB-001–PB-013 compliance playbooks |
| 5 | `evaluate_risk_decision_rules` | Evaluate RULE-001–RULE-012 to determine compliance/risk review flags |
| 6 | `compile_final_assessment` | Assemble and return full structured assessment JSON |

#### Questionnaire Logic Reference

| ID | Trigger Condition |
|---|---|
| **Q-001** | Standard questionnaire — always required for applicable risk levels |
| **Q-002** | AML/KYC — Medium/High risk financial industries |
| **Q-003** | High-risk industry questionnaire — High risk merchants only |
| **Q-004** | TPB questionnaire — triggered when TPB review is required |
| **Q-005** | PCI DSS — merchants processing card data at Medium/High risk |
| **Q-006** | Chargeback history — chargeback rate > 1.0% |
| **Q-007** | Healthcare/Pharmacy — pharma, health, or medical industry |
| **Q-008** | Gambling — gambling, betting, or casino industry |
| **Q-009** | Crypto — crypto, digital assets, or NFT industry |
| **Q-010** | High volume financial stability — annual volume > $1,000,000 |

#### Output (Flattened for Pipeline)

```json
{
  "risk_level": "HIGH | MEDIUM | LOW",
  "risk_score": 0,
  "high_risk_flag": false,
  "compliance_review_required": false,
  "risk_review_required": false,
  "tpb_threshold_exceeded": false,
  "required_questionnaires": ["Q-001", "Q-003"],
  "applicable_playbooks": ["PB-001", "PB-013"],
  "questionnaire_details": [
    { "id": "...", "name": "...", "description": "...", "link": "..." }
  ],
  "playbook_details": [
    { "id": "...", "name": "...", "framework": "...", "link": "..." }
  ],
  "overall_decision": "ESCALATE | REVIEW | APPROVE",
  "total_rules_triggered": 0,
  "assessment_date": "ISO-8601",
  "raw_output": "LLM output for audit"
}
```

---

### 3.4 Agent 4 — Document Requirements & Decision Logic Agent

> **File:** `agent_4_document_req_decision_logic_agent.py`

Determines the exact document requirements for a merchant based on their risk profile and compliance exposure. Uses a SQLite merchant history database (seeded with 30 real-world merchants) to distinguish new from existing merchants, and applies PD score tiering to calibrate document requirements.

#### Decision Logic — Document Tiering by PD Score

```
Risk Review = NO
    → No documents from risk side

Risk Review = YES, merchant NOT in DB
    → New Merchant, Full Review:
      Full financials, Bank statement, Interim financials, Processing statements

Risk Review = YES, merchant IN DB:
    PD <= 2.00%  → Light Review:
                   PD check, Questionnaire, Playbook link

    PD <= 5.00%  → Medium Review:
                   PD check, DDD documents, Questionnaire, Playbook link

    PD >  5.00%  → Full Review:
                   Full financials, Bank statement, Interim, Processing statements
```

#### Compliance Industry Check

Uses GPT-4o-mini to match the merchant's industry against 15 high-risk verticals defined in `compliance_requirements.md` (Financial Services, Cryptocurrency, Gambling, Adult Entertainment, Pharmaceuticals, Insurance, Forex, Precious Metals, Travel, MLM, Crowdfunding, Gaming, Alcohol/Tobacco, Digital Lending, Data Brokerage). Returns industry-specific artifact requirements.

#### Seeded Merchant Database (30 Records)

The database is pre-seeded with global enterprises across industries for demo and testing purposes. Production deployments should replace the seed function with a BigQuery or data warehouse integration.

| Merchant | Industry | PD Score | Review Tier |
|---|---|:---:|:---:|
| Amazon / Apple / Microsoft | E-commerce / Technology | 0.8–1.5% | Light |
| PayPal | Fintech | 4.00% | Medium |
| Shopify | E-commerce | 5.00% | Medium |
| Stripe | Fintech | 7.00% | Full |
| Uber / Airbnb | Mobility / Hospitality | 5.5–6.0% | Full |
| Tesla / Netflix | Automotive / Entertainment | 2.5–3.0% | Medium |

---

### 3.5 Agent 5 — Negative News & Adverse Media Detection Agent

> **File:** `agent_5_negative_news_detector.py` | **Class:** `Agent5NegativeNewsDetector`

Performs automated adverse media screening by querying multiple public data sources for negative news, regulatory actions, legal proceedings, financial distress signals, and reputational risks associated with the merchant. Uses GPT-4o-mini to classify and score findings.

#### Data Sources

- **Google News RSS feed** — news articles, press releases
- **DuckDuckGo News API** — independent news aggregation
- **Bing News Search** — Microsoft news index
- **GDELT Project** — global event database and news monitoring
- **Consumer Complaint portals** — customer grievance data

#### Risk Signal Detection

Screens for 50+ adverse signal keywords across categories including: fraud, scam, lawsuit, penalty, violation, breach, bankruptcy, criminal, indictment, investigation, sanction, money laundering, embezzlement, class action, regulatory enforcement, settlement, hack, and more.

#### Finding Classification

| Category | Severity Levels | Risk Score Range |
|---|---|---|
| **FINANCIAL** | CRITICAL / HIGH / MEDIUM / LOW | Score 76-100: CRITICAL → `DECLINE` |
| **LEGAL** | CRITICAL / HIGH / MEDIUM / LOW | Score 51-75: HIGH → `MANUAL_REVIEW` |
| **REGULATORY** | HIGH / MEDIUM / LOW | Score 26-50: MEDIUM → `EDD` |
| **REPUTATIONAL** | MEDIUM / LOW | Score 1-25: LOW → `APPROVE` with monitoring |
| **OPERATIONAL** | MEDIUM / LOW | Score 0: NONE → `APPROVE` |
| **SANCTIONS** | CRITICAL | Auto-escalate to `DECLINE` |

#### Industry-Specific Queries

Beyond standard screening, the agent runs industry-specific queries. For example: Fintech merchants trigger `compliance failure` and `payment fraud` queries; Gambling merchants trigger `illegal gambling`, `gaming license revoked`, `betting fraud`, and `gambling ban` queries.

#### Output Structure

```json
{
  "overall_risk_severity": "NONE | LOW | MEDIUM | HIGH | CRITICAL",
  "overall_risk_score": 0.0,
  "recommended_action": "APPROVE | EDD | MANUAL_REVIEW | DECLINE",
  "data_sufficiency": "SUFFICIENT | LIMITED | INSUFFICIENT",
  "findings": [
    { "title": "...", "category": "...", "severity": "...",
      "summary": "...", "source_url": "...", "confidence": 0.0 }
  ],
  "findings_by_category": { "LEGAL": 2, "REGULATORY": 1 },
  "corroboration_score": 0.0,
  "cross_source_summary": "string",
  "key_risk_indicators": ["string"],
  "executive_summary": "string",
  "search_queries_used": ["string"],
  "processing_time_seconds": 0.0
}
```

---

### 3.6 Agent 6 — Customer Sentiment Analysis Agent

> **File:** `agent_6_sentiment_analyzer.py` | **Class:** `Agent6CustomerSentimentAnalyzer`

Performs multi-source customer sentiment analysis using a hybrid approach combining rule-based (TextBlob), transformer-based (HuggingFace), and LLM-powered (GPT-4o-mini via LangChain) analysis. Reviews are collected from public sources and clustered using TF-IDF and K-Means.

#### Sentiment Analysis Stack

- **TextBlob** — rule-based polarity scoring (`-1.0` to `1.0`) for fast baseline
- **HuggingFace `AutoModelForSequenceClassification`** — transformer fine-tuned for sentiment
- **GPT-4o-mini (LangChain tool)** — contextual multi-review batch analysis with theme extraction
- **TF-IDF + K-Means** — review clustering to identify dominant sentiment themes

#### Tools (LangChain Agent)

- `analyze_sentiment_batch` — GPT-4o-mini batch review classification with confidence scores
- `extract_sentiment_themes` — identifies positive and negative recurring themes
- `calculate_final_sentiment_score` — aggregates across all models into a weighted score
- `compile_sentiment_report` — produces the final structured output

#### Output Schema

```json
{
  "overall_sentiment_score": 0.0,
  "customer_satisfaction_rating": 0.0,
  "recommendation_flag": "APPROVE | REVIEW | REJECT",
  "review_count": 0,
  "sentiment_distribution": { "positive": 0.6, "negative": 0.2, "neutral": 0.2 },
  "key_positive_themes": ["string"],
  "key_negative_themes": ["string"],
  "model_confidence": 0.0
}
```

> **Important Note**
> Agent 6 requires PyTorch 2.2.2 and the HuggingFace transformers library. On first run, transformer models are downloaded and cached locally to `HF_HOME` (defaults to a `hf_cache_agent6` directory in the working directory). Subsequent runs use the local cache. Telemetry and progress bars are disabled by default for production use.

---

### 3.7 Agent 7 — Orchestration Agent

> **File:** `agent_7_orchestration_agent.py` | **Class:** `Agent7OrchestrationAgent` | **Framework:** LangGraph StateGraph

The central decision-making engine. Implements a LangGraph state machine that ingests outputs from Agents 3, 4, 5, and 6, detects inter-agent conflicts, applies policy rules, and produces the final onboarding decision with full rationale and audit trail. Both conflict detection and final decision logic are LLM-powered with rule-based fallbacks.

#### LangGraph Node Sequence

| Node | Name | Action |
|:---:|---|---|
| 1 | **INTAKE** | Normalize all agent inputs, validate field presence, initialize orchestration state |
| 2 | **ASSESS** | Evaluate each agent signal independently — compute risk flags, severity scores, compliance flags |
| 3 | **CONFLICT** | LLM detects disagreements between agents (e.g., Agent 3 says LOW, Agent 5 says CRITICAL) |
| 4 | **DECIDE** | LLM applies policy rules to resolve conflicts and produce `APPROVE` / `MANUAL_REVIEW` / `REJECT` |
| 5 | **FINALIZE** | Assemble decision rationale, escalation reasons, required actions, audit metadata |

#### Decision Logic (Priority-Ordered)

- **REJECT triggers:** CRITICAL adverse media, unresolvable compliance failures, invalid validation
- **MANUAL_REVIEW triggers:** HIGH adverse media, compliance review required, high risk flag, sentiment REJECT, agent conflicts
- **APPROVE triggers:** LOW/MEDIUM risk, no adverse media, positive sentiment, all questionnaires assignable

#### Input Dataclasses

```python
Agent3Output:  merchant_name, risk_level, high_risk_flag, tpb_threshold_exceeded,
               compliance_review_required, required_questionnaires, applicable_playbooks

Agent4Output:  merchant_name, pd_score, documents_required, merchant_type,
               risk_review, high_risk_match, artifacts_required

Agent5Output:  merchant_name, overall_risk_severity, overall_risk_score,
               recommended_action, data_sufficiency, findings_count, corroboration_score

Agent6Output:  merchant_name, overall_sentiment_score, customer_satisfaction_rating,
               recommendation_flag, review_count, negative_ratio
```

#### Output Schema

```json
{
  "final_decision": "APPROVE | MANUAL_REVIEW | REJECT",
  "decision_rationale": "string",
  "escalation_reasons": ["string"],
  "required_actions": ["string"],
  "merchant_name": "string",
  "audit_id": "SHA-256 hash for traceability",
  "timestamp": "ISO-8601"
}
```

---

### 3.8 Agent 8 — Summarization Agent

> **File:** `agent_8_summarization_agent.py`

Generates a professional, narrative-style merchant onboarding report from all upstream agent outputs. Designed to serve both Sales teams (who need actionable document collection guidance) and Risk & Compliance teams (who need a policy-aligned assessment narrative). Uses GPT-4o-mini with a detailed system prompt enforcing strict output structure.

#### Report Sections (Mandatory Output Structure)

1. **Executive Overview** — High-level summary with recommendation-style conclusion
2. **Merchant Snapshot** — Identity, industry, website, merchant type, MCC
3. **Data Validation Status** — Quality score, completeness, sales implications
4. **Risk & Compliance Overview** — Risk level, review requirements, sales-friendly interpretation
5. **Financial & Documentation Requirements** — PD score, merchant type, required documents (CRITICAL)
6. **Adverse Media Considerations** — Findings summary, sales awareness notes
7. **Customer Sentiment Overview** — Score, rating, interpretation
8. **Overall Risk Positioning** — Strengths, considerations, balanced view
9. **Recommended Next Steps** — Immediate actions, coordination, follow-up
10. **Summary for Sales** — 3–5 line takeaway emphasizing opportunity viability

#### Critical Design Constraints

- **NEVER** outputs a hard decision (no `APPROVE` or `REJECT` — uses advisory language only)
- Agent 4 details (PD score, merchant type, documents) must be preserved **EXACTLY**
- Advisory phrases: *"This profile supports onboarding subject to…"*, *"A cautious approach may be warranted…"*
- Risk priority: **Adverse Media > Sentiment > Regulatory > Risk Score > Financial Profile**
- Input is filtered to top 5 adverse media findings to avoid token overflow

---

## 4. Pipeline Orchestrator

> **File:** `pipeline.py` | **Class:** `MerchantOnboardingPipeline`

The `pipeline.py` module is the production entry point for the full multi-agent workflow. It manages sequential and parallel agent execution, error isolation, output normalization, and final report assembly.

### 4.1 Execution Sequence

1. **Agent 2** (Validation) → sequential, can terminate early
2. **Agents 3, 5, 6** → parallel (`ThreadPoolExecutor`, `max_workers=3`)
3. **Agent 4** → sequential, after Agent 3
4. **Agent 7** (Orchestration) → sequential, after all above
5. **Agent 8** (Summarization) → sequential, final step

### 4.2 Error Isolation

Each agent is wrapped in an exception handler. If any parallel agent (3, 5, or 6) fails, a default empty output is substituted and the pipeline continues. This ensures a failed adverse media lookup or sentiment crawl does not abort the entire assessment. **Agent 2 failure is the only hard-stop condition.**

### 4.3 Final Report Structure

```json
{
  "merchant_name": "string",
  "pipeline_timestamp": "ISO-8601 datetime",
  "pipeline_duration_seconds": 0.0,
  "final_decision": "APPROVE | MANUAL_REVIEW | REJECT",
  "decision_rationale": "string",
  "escalation_reasons": ["string"],
  "required_actions": ["string"],
  "agent_outputs": {
    "agent_2_validation": { "...": "..." },
    "agent_3_risk": { "...": "..." },
    "agent_4_documents": { "...": "..." },
    "agent_5_adverse_media": { "...": "..." },
    "agent_6_sentiment": { "...": "..." },
    "agent_7_orchestration": { "...": "..." },
    "agent_8_summary": { "summary_text": "string" }
  },
  "pipeline_trace": ["timestamped execution log entries"]
}
```

---

## 5. Web UI & API Server

> **Files:** `server.py` + `index.html`

The platform ships with a real-time monitoring web interface that displays agent execution status, streams live logs via Server-Sent Events (SSE), and provides drill-down inspection of each agent's JSON output.

### 5.1 Server Architecture

- **Framework:** Flask 3.0+ with Flask-CORS for cross-origin support
- **Streaming:** Server-Sent Events (SSE) for real-time push from server to browser
- **Threading:** Each pipeline run executes in a daemon thread; events are queued and streamed
- **Static files:** `index.html` served from the `ui/` directory

### 5.2 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the main web UI (`index.html`) |
| `POST` | `/run` | Start a pipeline run; returns SSE stream of agent events |
| `GET` | `/health` | Health check endpoint; returns `{status: 'ok'}` |

### 5.3 SSE Event Schema

The `/run` endpoint streams newline-delimited JSON events. Frontend clients parse these to update agent card states in real-time.

```javascript
// Pipeline start
{ "type": "pipeline_start", "timestamp": "...", "merchant_name": "..." }

// Agent started
{ "type": "agent_start", "timestamp": "...", "agent_id": 1, "agent_name": "..." }

// Agent completed
{ "type": "agent_complete", "timestamp": "...", "agent_id": 1,
  "agent_name": "...", "output": { "...": "agent output dict" } }

// Agent errored
{ "type": "agent_error", "timestamp": "...", "agent_id": 1, "error": "..." }

// Pipeline finished
{ "type": "pipeline_complete", "timestamp": "...",
  "final_decision": "...", "summary": "..." }
```

### 5.4 UI Features

| Feature | Description |
|---|---|
| **Demo Merchant Buttons** | PayPal, BetKing, ShopEasy, Bistro — pre-fill form with sample data |
| **Manual Entry Form** | Fields: Merchant Name, Address, Website, Industry, MCC |
| **Real-time Agent Cards** | 8 cards showing idle / running / complete / error states |
| **Agent Output Drill-down** | Click any completed card to see summary and raw JSON output |
| **Final Decision Panel** | `APPROVE` / `MANUAL_REVIEW` / `REJECT` with decision rationale |
| **Live Log Stream** | Timestamped execution trace at the bottom of the page |
| **Agent State Indicator** | Pulsing blue dot for running, green check for complete, red X for error |

---

## 6. Installation & Setup

### 6.1 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Required — agents use `match/case` and 3.11 typing features |
| pip | Latest | Upgrade with: `pip install --upgrade pip` |
| OpenAI API Key | Active subscription | Set as `OPENAI_API_KEY` in `.env` file |
| virtualenv | Optional | Recommended for dependency isolation |
| RAM | 8 GB minimum | PyTorch + transformer models require substantial memory |
| Disk Space | ~5 GB | HuggingFace model cache on first run |

### 6.2 Environment Setup

```bash
# Step 1: Clone the repository
git clone <repository-url>
cd merchant-onboarding

# Step 2: Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate          # Mac / Linux
# venv\Scripts\activate           # Windows

# Step 3: Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Configure API key
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-...
```

### 6.3 Directory Structure

```
merchant-onboarding/
├── agents/
│   ├── agent_1_data_ingestion.py
│   ├── agent_2_validation_agent.py
│   ├── agent_3_risk_compliance.py
│   ├── agent_4_document_req_decision_logic_agent.py
│   ├── agent_5_negative_news_detector.py
│   ├── agent_6_sentiment_analyzer.py
│   ├── agent_7_orchestration_agent.py
│   ├── agent_8_summarization_agent.py
│   └── pipeline.py
├── merchant-ui/
│   ├── server.py
│   └── ui/
│       └── index.html
├── data/
│   ├── policy_data.xlsx          # Policy rules (required for Agents 2, 3)
│   ├── merchants.xlsx            # Merchant details
│   └── merchants.db              # Auto-created by Agent 1
├── docs/
│   └── compliance_requirements.md  # Auto-created by Agent 4
├── requirements.txt
└── .env                          # API keys (not committed to VCS)
```

### 6.4 Policy Data File

The system requires a `policy_data.xlsx` file at `data/policy_data.xlsx`. This file must contain the following Excel sheets:

- **`Industry_Risk_Mapping`** — Columns: `Industry`, `MCC_Code`, `Risk_Level`, `Risk_Score`, `High_Risk_Flag`, `Risk_Reason`
- **`TPB_Thresholds`** — Columns: `Risk_Level`, `TPB_Percentage_Min`, `TPB_Percentage_Max`, `Action_Required`, `TPB_Review_Notes`
- **`Questionnaire_Templates`** — Columns: `Questionnaire_ID`, `Questionnaire_Name`, `Description`, `Applicable_Risk_Levels`, `Applicable_Industries`, `Link`
- **`Compliance_Playbooks`** — Columns: `Playbook_ID`, `Playbook_Name`, `Regulatory_Framework`, `Risk_Level`, `Applicable_Industries`, `Playbook_Link`
- **`Risk_Decision_Logic`** — Columns: `Rule_ID`, `Rule_Name`, `Condition`, `Action`

> **Note**
> Without `policy_data.xlsx`, Agents 2 and 3 fall back to hardcoded defaults. For production use, populate this file from your institution's risk policy database.

---

## 7. Running the System

### 7.1 Run Individual Agents (Testing)

```bash
# Run standalone agent tests (each agent has a __main__ block)
python agents/agent_1_data_ingestion.py
python agents/agent_2_validation_agent.py
python agents/agent_3_risk_compliance.py
python agents/agent_4_document_req_decision_logic_agent.py
```

### 7.2 Run Full Pipeline (Headless)

```bash
# Edit pipeline.py to set your merchant input, then run:
cd agents
python pipeline.py

# Output: onboarding_<merchant_name>_report.json
```

### 7.3 Run Web UI

```bash
# Start the UI server (default port 5050)
python merchant-ui/server.py
```

You should see:

```
============================================================
    Merchant Onboarding Pipeline UI Server
    http://localhost:5050
    Agents dir: /path/to/agents
============================================================
```

To run on a custom port:

```bash
PORT=8080 python merchant-ui/server.py
```

### 7.4 Using the Web UI

1. Open browser at `http://localhost:5050`
2. Click a demo button (PayPal, BetKing, ShopEasy, Bistro) or fill in the form manually
3. Click **Run Pipeline** — agent cards begin activating in real time
4. Watch the log stream at the bottom for detailed execution trace
5. Click any completed (green) agent card to view its output
6. Final decision (`APPROVE` / `MANUAL_REVIEW` / `REJECT`) appears in the left panel

### 7.5 Programmatic Pipeline Usage

```python
from pipeline import MerchantOnboardingPipeline

pipeline = MerchantOnboardingPipeline()  # reads OPENAI_API_KEY from env

result = pipeline.run({
    'merchant_name': 'BetKing Online Ltd.',
    'address': '123 Main St',
    'website': 'https://m.betking.com/',
    'industry': 'Online Gambling / Betting',
    'mcc': '7995',
})

print(result['final_decision'])    # APPROVE | MANUAL_REVIEW | REJECT
print(result['decision_rationale'])
```

---

## 8. Dependencies

### 8.1 Full `requirements.txt` Reference

| Package | Version | Used By |
|---|---|---|
| `requests` | `>=2.31.0` | Agent 5 (news fetching), Agent 6 (review scraping) |
| `beautifulsoup4` | `>=4.12.0` | Agent 5 & 6 (HTML parsing for scraped content) |
| `pandas` | latest | Agents 2, 3, 4 (Excel/DataFrame policy loading) |
| `textblob` | latest | Agent 6 (rule-based sentiment baseline) |
| `openpyxl` | latest | Agents 1, 2, 3 (Excel read/write) |
| `openai` | `>=1.30.0` | Agents 3, 5, 6, 8 (GPT-4o-mini API calls) |
| `python-dotenv` | latest | All agents (`OPENAI_API_KEY` from `.env`) |
| `pdfplumber` | latest | Document parsing utility |
| `langchain` | `0.2.14` | Agent 3, 6, 7 (tool orchestration) |
| `langchain-openai` | `0.1.22` | Agent 3, 6, 7 (`ChatOpenAI` integration) |
| `langchain-core` | `>=0.2.23` | Agent 4 (`PromptTemplate`) |
| `langgraph` | `0.2.16` | Agent 7 (`StateGraph` orchestration) |
| `torch` | `2.2.2` | Agent 6 (HuggingFace transformer inference) |
| `transformers` | `4.38.2` | Agent 6 (`AutoModel` sentiment classifier) |
| `scikit-learn` | latest | Agent 6 (TF-IDF, K-Means clustering) |
| `tensorflow` | latest | Agent 6 (additional ML backend) |
| `flask` | `>=3.0.0` | `server.py` (web UI and API endpoints) |
| `flask-cors` | `>=4.0.0` | `server.py` (cross-origin request support) |

---

## 9. Configuration & Environment Variables

### 9.1 Required Environment Variables

| Variable | Required | Description |
|---|:---:|---|
| `OPENAI_API_KEY` | **YES** | OpenAI API key for GPT-4o-mini (Agents 3, 5, 6, 7, 8) |
| `PORT` | No (default: `5050`) | Port for the Flask web UI server |

### 9.2 Per-Agent Configuration

| Agent | Config Variable | Notes |
|:---:|---|---|
| Agent 1 | `db_path = 'merchants.db'` | SQLite file path; override in constructor |
| Agent 2 | `policy_xlsx` (optional) | Pass path to policy Excel file for full taxonomy |
| Agent 3 | `POLICY_FILE = 'data/policy_data.xlsx'` | Edit constant at top of file |
| Agent 3 | `MODEL_NAME = 'gpt-4o-mini'` | LLM model for risk assessment |
| Agent 4 | `DB_PATH = 'merchant.db'` | Separate DB for PD score history |
| Agent 4 | `COMPLIANCE_MD_PATH` | Path to `compliance_requirements.md` |
| Agent 5 | `openai_api_key` | Passed to constructor from environment |
| Agent 6 | `HF_HOME` env var | HuggingFace cache directory for transformers |
| Agent 7 | `gpt-4o-mini`, `temperature=0` | Deterministic LLM for orchestration |
| Agent 8 | `max_tokens=1500` | Summary report length limit |

---

## 10. Troubleshooting

### 10.1 Common Issues & Resolutions

| Issue | Resolution |
|---|---|
| **`OPENAI_API_KEY` not set / `AuthenticationError`** | Ensure `.env` file exists with `OPENAI_API_KEY=sk-...` and `python-dotenv` is installed. Verify key is active at platform.openai.com |
| **`No module named 'flask'`** | Run: `pip install flask flask-cors` (ensure venv is activated) |
| **`No module named 'langchain'`** | Run: `pip install langchain==0.2.14 langchain-openai==0.1.22 langgraph==0.2.16` |
| **Port 5050 already in use** | Run server on alternative port: `PORT=8080 python merchant-ui/server.py` |
| **`policy_data.xlsx` not found** | Agents 2 and 3 use fallback defaults. For full functionality, place policy Excel at `data/policy_data.xlsx` |
| **Agent 6: Slow on first run** | HuggingFace downloads transformer models (~1-2 GB). Subsequent runs use local cache at `HF_HOME` |
| **Agent 3 returns `raw_output` only** | The pipeline's `_parse_agent3_output` regex extractor handles this. Verify `policy_data.xlsx` has all 5 required sheets |
| **Agent 5: No findings returned** | The news APIs may rate-limit or return empty results. `data_sufficiency` will be `LIMITED` and score defaults to 0 (APPROVE-leaning) |
| **Agents fail silently in UI** | Check browser console for SSE errors and terminal logs from `server.py`. Increase logging verbosity |
| **SQLite locked error** | Close any external DB browser. Agent 1 uses connection-per-operation pattern which should prevent this |
| **Transformer model OOM error** | Reduce batch sizes in Agent 6 or upgrade to a machine with >= 16 GB RAM. Disable TensorFlow: uninstall `tensorflow` from `requirements.txt` |
| **Agent 7 returns `MANUAL_REVIEW` unexpectedly** | This is expected when agents disagree — check `agent_outputs` in the final report JSON for conflict details |

### 10.2 Debug Mode

Agent 3 runs with `verbose=True` in the `AgentExecutor`, printing every LangChain tool invocation to stdout. This is useful for debugging policy rule evaluation. To silence it, set `verbose=False` in the `agent_executor` initialization.

### 10.3 Agent-Level Testing

Every agent module contains a `__main__` block with test cases that can be run independently. This allows isolated testing of individual components without running the full pipeline. Test cases cover low-risk, medium-risk, high-risk, and incomplete data scenarios.

```bash
# Test each agent independently
python agents/agent_2_validation_agent.py   # 3 test scenarios
python agents/agent_3_risk_compliance.py    # Restaurant low-risk example
python agents/agent_4_document_req_decision_logic_agent.py  # Amazon + BetKing
```

---

## 11. Security & Compliance Considerations

### 11.1 Data Handling

- Merchant data is persisted in local SQLite databases — not transmitted to third parties beyond the OpenAI API
- All OpenAI API calls transmit merchant name, website, and industry for analysis — ensure your OpenAI data usage agreements cover this
- The system does not store API keys in code — always use environment variables via `.env`
- Public web scraping in Agents 5 and 6 is read-only and does not create accounts or submit forms

### 11.2 Decision Auditability

- Every pipeline run produces a full JSON report with timestamped `pipeline_trace`
- Agent 7 generates an `audit_id` (SHA-256 hash) for each decision for traceability
- All agent intermediate outputs are preserved in the `agent_outputs` section of the final report
- Agent 8 never states a hard decision — all outputs use advisory language to preserve human oversight

### 11.3 Production Hardening Checklist

- [ ] Replace SQLite with PostgreSQL or a managed database for concurrent access
- [ ] Replace the seed merchant database with a BigQuery / data warehouse integration for real PD scores
- [ ] Add rate limiting to the Flask API (`/run` endpoint)
- [ ] Implement authentication (OAuth 2.0 or API key headers) on the web server
- [ ] Store OpenAI API key in a secrets manager (AWS Secrets Manager, HashiCorp Vault) rather than `.env`
- [ ] Enable HTTPS on the Flask server in production (use a reverse proxy like nginx)
- [ ] Log all decisions to a compliance audit table with merchant ID, timestamp, decision, and agent outputs
- [ ] Implement a human review queue for all `MANUAL_REVIEW` decisions

---

<div align="center">

**Agentic AI for Merchant Onboarding**
*Technical Documentation v1.0.0 — Internal Use Only*

</div>