# Setup Instructions

## Agentic AI for Merchant Onboarding

---

## Prerequisites

Ensure the following are installed on your system:

* **Python 3.11+**
* **pip (Python package manager)**
* (Optional) **virtualenv**

Verify Python installation:

```bash
python3.11 --version
```

---

## Environment Setup

### 1. Create Virtual Environment

```bash
python3.11 -m venv venv
```

### 2. Activate Virtual Environment

**Mac / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

---

## Install Dependencies

Upgrade pip and install required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Project

### Run Individual Agents

You can execute agents independently for testing:

```bash
python agents/agent_2_validation_agent.py
```

---

### Run Full Pipeline

To execute the complete multi-agent workflow:

```bash
python agents/pipeline.py
```

---

### Run UI Locally

Start the UI server:

```bash
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

---

## Access the Application

Open your browser and navigate to:

```
http://localhost:5050
```

---

# UI Application Usage

* Load a demo merchant using top buttons (**PayPal, BetKing, ShopEasy, Bistro**)
* Or manually enter merchant details
* Click **"Run Pipeline"**
* Agents execute and update in real time
* Click any completed agent card (green) to view:
  * Summary view
  * Raw JSON output
* Final decision (**APPROVE / REJECT / MANUAL_REVIEW**) appears in the left panel
* Log stream at bottom shows real-time execution events

---

# Agent Cards (UI Behavior)

| State        | Description                   |
| ------------ | ----------------------------- |
| **Idle**     | Grey border, grey badge       |
| **Running**  | Blue border, pulsing blue dot |
| **Complete** | Click to expand output        |
| **Error**    | Red border with error details |

---

## Agents Execution Flow

* **Agents 3, 5, 6** → run in parallel
* **Agent 4** → starts after Agent 3
* **Agent 7** → starts after Agents 3, 4, 5, 6 complete

---

# Changing the Port

Run server on a different port:

```bash
PORT=8080 python merchant-ui/server.py
```

---

# 🛠️ Troubleshooting

| Issue                    | Solution                                   |
| ------------------------ | ------------------------------------------ |
| OPENAI_API_KEY not set   | Ensure `.env` file contains API key        |
| No module named 'flask'  | `pip install flask flask-cors`             |
| Agents fail silently     | Check browser log stream and terminal logs |
| Port 5050 already in use | Use `PORT=8080`                            |

---
