# 🔍 Agentic Company Auditor

> An agentic AI system that researches any company in real-time and generates a structured Data/AI consulting audit — powered by **LangGraph**, **Gemini**, and **FastAPI**.

---

## What It Does

Send a company name. Get a full consulting audit streamed back in real-time.

The system autonomously researches the company on the web, validates the quality of its findings, proposes two Data/AI use cases tailored to the company's actual challenges, and delivers a structured JSON report — all without human intervention.

```bash
POST /api/v1/audit
{"company_name": "ONCF"}

# Streams back:
# → research node: real company context from the web
# → validation node: quality gate (retries if context is insufficient)
# → ideation node: 2 AI/Data use cases grounded in real challenges
# → reporting node: structured Pydantic-validated JSON report
```

---

## Architecture

The system is built as a **stateful directed graph** using LangGraph. Each node is a discrete, testable unit of work. The graph manages state transitions explicitly — this is not a chain of LLM calls, it is a controllable workflow with branching logic.

```
[research_node] ──► [validation_node] ──► [ideation_node] ──► [reporting_node]
                          │
                          └── (retry if context < 3 facts, max 2 retries)
                                    │
                                    ▼
                             [research_node]
```

### State Schema (`TypedDict`)

```python
class AuditState(TypedDict):
    company_name: str
    company_context: str        # populated by research_node
    proposed_use_cases: List[str]  # populated by ideation_node
    final_report: str           # Pydantic-validated JSON, populated by reporting_node
    retry_count: int            # managed by the conditional edge
```

### Nodes

| Node | Role | Tool |
|---|---|---|
| `research_node` | Web search + LLM synthesis | DuckDuckGo + Gemini |
| `validation_node` | Quality gate — counts distinct facts | Pure Python |
| `ideation_node` | Proposes 2 grounded AI/Data use cases | Gemini |
| `reporting_node` | Structures output as a validated JSON report | Gemini + Pydantic |

### Conditional Edge

`should_retry` inspects the state after `validation_node`. If `company_context` contains fewer than 3 distinct facts and `retry_count < 2`, the graph loops back to `research_node`. This prevents hallucinated or thin research from propagating downstream.

```python
def should_retry(state: AuditState) -> str:
    facts = _extract_distinct_facts(state["company_context"])
    if len(facts) >= 3:
        return "continue"
    if state["retry_count"] < 2:
        return "retry"
    return "continue"
```

---

## Architecture Decision: Why LangGraph over CrewAI

| Criterion | LangGraph | CrewAI |
|---|---|---|
| **Control flow** | Explicit `StateGraph` with typed edges and conditional branching | Abstracted behind "crew" and "task" objects |
| **Testability** | Each node is a plain Python function — unit-testable in isolation | Agents are tightly coupled, harder to mock |
| **State management** | Typed `TypedDict` shared across all nodes | Implicit, managed by the framework |
| **Observability** | `astream_events()` exposes every node transition in real-time | Limited event-level visibility |
| **Production fit** | Designed for stateful, long-running workflows | Better suited for rapid prototyping |

LangGraph forces you to think in terms of **Software Engineering**: define your state, define your nodes, define your transitions. The graph is the architecture documentation.

---

## Output Schema

The `reporting_node` produces a Pydantic-validated `AuditReport`:

```python
class AuditReport(BaseModel):
    company_name: str
    company_context: str
    proposed_use_cases: List[UseCase]  # title, description, expected_value
    assumptions: List[str]
    next_steps: List[str]
```

If the LLM returns malformed JSON, `model_validate_json()` raises a `ValidationError` — caught and surfaced to the client via SSE. The report is never silently corrupted.

---

## Streaming API

The endpoint uses **Server-Sent Events** via FastAPI's `StreamingResponse`. Each node emits only its relevant output key — the full state is never repeated.

```
data: {"node": "research",    "output": {"company_context": "..."}}
data: {"node": "validation",  "output": {"retry_count": 0}}
data: {"node": "ideation",    "output": {"proposed_use_cases": [...]}}
data: {"node": "reporting",   "output": {"final_report": "{...}"}}
```

This design makes the API directly consumable by any frontend without additional parsing.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 0.2+ |
| LLM | Google Gemini Flash Latest (`langchain-google-genai`) |
| Web Search | DuckDuckGo (`langchain-community`) |
| Output Validation | Pydantic v2 |
| API | FastAPI + StreamingResponse (SSE) |
| Testing | pytest + `TestClient` |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
agentic-company-auditor/
├── graph.py          # LangGraph StateGraph — nodes, edges, conditional logic
├── main.py           # FastAPI app — SSE endpoint, lifespan, health check
├── test_api.py       # pytest — endpoint tests with mocked LLM
├── Dockerfile        # python:3.11-slim, non-root user
├── docker-compose.yml
├── requirements.txt
└── .env.example      # GOOGLE_API_KEY template (not committed)
```

---

## Getting Started

### Local

```bash
# 1. Clone and install
git clone https://github.com/medboukechouch/agentic-company-auditor
cd agentic-company-auditor
pip install -r requirements.txt

# 2. Set your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Run
uvicorn main:app --reload

# 4. Test
curl -X POST http://localhost:8000/api/v1/audit \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Converteo"}' \
  --no-buffer
```

### Docker

```bash
docker compose up --build
```

### Run Tests

```bash
pytest test_api.py -v
```

---

## Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Example Output

**Input:** `{"company_name": "ONCF"}`

**Streamed output:**

```
research  → Real web-sourced context: Al Boraq HSR, Infraway-Maroc, oncf-voyages platform
ideation  → Use case 1: Predictive Maintenance (IoT + ML on rail infrastructure)
            Use case 2: Dynamic Pricing & Demand Forecasting (oncf-voyages data)
reporting → Structured JSON: company_context, use_cases, assumptions, next_steps
```

The system correctly identified ONCF as Morocco's state-owned rail operator and grounded both use cases in its actual subsidiaries and data assets — without any manual input.

---

## Author

**Mohamed Boukechouch**
Master CSSD — Cybersécurité et Sciences des Données, Paris 8
[linkedin.com/in/mohamed-boukechouch](https://linkedin.com/in/mohamed-boukechouch) · [github.com/medboukechouch](https://github.com/medboukechouch)