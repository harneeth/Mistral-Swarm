# Mistral Swarm

> A multi-agent desktop automation system built on the **Mistral AI** platform.
> Only Mistral models and agents are used throughout.

---

## Overview

Mistral Swarm is a collaborative hackathon project that automates computer tasks using a pipeline of seven specialised AI agents. A user states their intent in plain language — for example, *"summarise my emails and draft a reply to John"* — and the system coordinates all agents to carry out the task autonomously on the desktop.

The system is designed to be modular: each module has a single responsibility and communicates with the others through well-defined APIs and a shared real-time dashboard.

---

## System architecture

The pipeline consists of seven modules that execute in sequence:

| # | Module | Responsibility |
|---|---|---|
| 1 | **Interaction** | Receives the user's intent from the terminal or UI |
| 2 | **Parser** | Breaks the intent into a structured list of steps |
| 3 | **Info Gathering** | Fetches additional context from the web; requests user permissions when needed |
| 4 | **Task Performer** | Translates steps into computer actions (mouse, keyboard, applications) |
| 5 | **Typing / Creation** | Generates text content (emails, summaries, replies) via a Mistral agent |
| 6 | **Overseer** | Orchestrates the full pipeline, handles errors and retries |
| 7 | **Scheduler / Init** | Manages scheduled tasks and system startup |

Each module runs as an independent FastAPI service. The **Overseer (Module 6)** coordinates execution and routes results between modules. All modules report their status to the **Terminal Dashboard** in real time.

---

## Modules in detail

### Module 1 — Interaction
Entry point for the user. Accepts plain-language commands typed at the terminal dashboard prompt and forwards them to the Parser.

### Module 2 — Parser
Converts the raw user intent into a structured task object: a list of named steps with associated parameters. The output is consumed by the Task Performer and the Overseer.

### Module 3 — Info Gathering
Performs web searches and data lookups to enrich the task context. Also handles permission prompts when an action requires explicit user approval.

### Module 4 — Task Performer
Executes concrete actions on the computer. Receives a structured task and translates each step into one or more low-level actions.

GUI element detection uses the Salesforce **GPA-GUI-Detector** model (a YOLO-based detector fine-tuned on GUI elements), loaded via Ultralytics and wrapped in `gui_detector.py`.

**Action vocabulary:**

| Action | Description |
|---|---|
| `MOVE_CURSOR` | Move the mouse pointer |
| `CLICK` / `DOUBLE_CLICK` | Click on a target element |
| `SCROLL` | Scroll a panel or list |
| `SELECT_TEXT` | Select text in a field |
| `FOCUS_INPUT` | Focus a UI element |
| `TYPE_TEXT` | Type text (calls Module 5 when text must be generated) |
| `OPEN_APP` | Launch an application |
| `OPEN_URL` | Open a URL in the browser |
| `PRESS_KEY` | Send a keyboard shortcut |
| `WAIT` | Pause execution |

**API (port 8001):**
```
POST /run_task
POST /scenario/summarize_emails
POST /scenario/draft_reply
```

### Module 5 — Typing / Content Creation
Generates written content on demand. Any module that needs text produced (an email draft, a summary, a reply) calls this module. It uses the **Mistral Content Writer agent** via the Mistral Agents API and returns a plain string.

**API (port 8000):**
```
POST /generate_text       { "task_description": "...", "context": "..." }
POST /summarize_emails    { "emails": [...] }
```

### Module 6 — Overseer
Central coordinator. Monitors all modules, routes results between them, handles failures, and decides when to retry or escalate. Also listens to the dashboard WebSocket to receive real-time user commands.

### Module 7 — Scheduler / Init
Handles scheduled and recurring tasks (e.g. "every morning, summarise my inbox"). Also responsible for system startup and initialising the other modules in the correct order.

---

## Terminal Dashboard

A real-time interface that visualises the state of all seven agents as the pipeline runs. Intended both as a demo interface and as a development tool for monitoring the system end-to-end.

```
┌─ MISTRAL SWARM v1.0 ──────── TOKENS: 0042 ── 2026-03-01 14:22:01 UTC ─┐
│ SYSTEM_LOG                   │ AGENT_MESH_TOPOLOGY                      │
│ [14:22:01] Boot OK           │  ┌──────────┐  ┌──────────┐             │
│ [14:22:02] Overseer ready    │  │● MOD_1   │  │● MOD_2   │             │
│ [14:22:05] > draft reply...  │  │  IDLE    │  │  IDLE    │             │
│ [14:22:05] [MOD_4] active    │  ├──────────┤  ├──────────┤             │
│ [14:22:06] [MOD_5] active    │  │◉ MOD_4   │  │◉ MOD_5   │             │
│                              │  │  BUSY    │  │  BUSY    │             │
│                              │  └──────────┘  └──────────┘             │
├──────────────────────────────┴─────────────────────────────────────────┤
│ ● USER@MISTRAL_SWARM:~$ █                                              │
└────────────────────────────────────────────────────────────────────────┘
```

Two interface options — same backend:

| | CLI (TUI) | Web |
|---|---|---|
| Command | `python tui.py` | `uvicorn dashboard:app --port 8080` |
| Access | Terminal window | Browser `localhost:8080` |
| Login | — | `admin` / `mistral2026` |

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/JIBLIR/Mistral_Operator.git
cd Mistral_Operator

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env            # then fill in your keys
```

**`.env` variables:**
```
MISTRAL_API_KEY=your_key_here
MISTRAL_AGENT_ID=your_agent_id_here

DASHBOARD_USER=admin
DASHBOARD_PASS=mistral2026
```

> `MISTRAL_AGENT_ID` refers to the shared Content Writer agent configured by the team in the Mistral console. Team members can substitute their own agent ID if needed.

---



## Integration — pushing events to the dashboard

Any module can send status updates to the dashboard.

### From a separate process (HTTP POST)

```python
import httpx

# Light up an agent card
httpx.post("http://localhost:8080/event", json={
    "type": "agent", "id": "agent4", "status": "busy", "task": "click_compose"
})

# Add a line to the system log
httpx.post("http://localhost:8080/event", json={
    "type": "log", "text": "[MOD_4] Moving cursor...", "cls": "busy"
})
```

### From the same Python process

```python
from dashboard import broadcast_event

await broadcast_event({"type": "agent", "id": "agent5", "status": "busy", "task": "generating_email"})
await broadcast_event({"type": "agent", "id": "agent5", "status": "idle"})
```

### Event schema reference

```jsonc
{ "type": "log",          "text": "...", "cls": "busy|success|error|system" }
{ "type": "agent",        "id": "agent4", "status": "busy", "task": "label" }
{ "type": "agent",        "id": "agent4", "status": "idle" }
{ "type": "tokens",       "count": 1842 }
{ "type": "tokens_delta", "delta": 38 }
{ "type": "input",        "text": "user command text" }
```

**Agent IDs:** `agent1` through `agent7` — map to MOD_1 … MOD_7.



## Tech stack

| Layer | Technology |
|---|---|
| AI / Agents | [Mistral AI](https://mistral.ai) — `mistral-large`, Content Writer agent |
| GUI detection | [Salesforce/GPA-GUI-Detector](https://huggingface.co/Salesforce/GPA-GUI-Detector) via [Ultralytics](https://ultralytics.com) |
| Backend | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) |
| Terminal UI | [Textual](https://textual.textualize.io) |
| Web UI | Vanilla HTML / CSS / JS (inline, no build step) |
| Real-time | WebSockets |
| HTTP client | [httpx](https://www.python-httpx.org) |

---

## Configuration and customisation

### Changing the Mistral agent

Create a new agent in the [Mistral console](https://console.mistral.ai), then update `.env`:
```
MISTRAL_AGENT_ID=ag_your_new_agent_id
```
The prompt template is in `agent5.py` → `ContentWriterClient._build_prompt()`.

### Dashboard settings

```
DASHBOARD_USER=admin
DASHBOARD_PASS=mistral2026
DASHBOARD_SECRET_KEY=...      # long random string recommended for production
```

Port is controlled by `SERVER_PORT` in `tui.py` and `dashboard.py` (default `8080`).

### Adding new action types

Add a value to the `ActionType` enum in `actions.py`, then handle it in `executor.py` → `BasicTaskPerformer.run_task()`.

---

NOTE: We could not complete Module 3 and Module 7 in time due to limited members and a lot of work. However, these are easy to implement.


## Limitations and future work

**Current limitations:**
- The GUI detector requires `torch` and `ultralytics` and benefits from a CUDA-capable GPU for real-time use.
- The dashboard has no persistent log — events are not saved between sessions.
---

## License and credits
A model "model.pt" was used with YOLO which was under the MIT License.

This project was developed collaboratively during the **Mistral AI Hackathon 2026**. All modules are the result of team work across multiple contributors.
