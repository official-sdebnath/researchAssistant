
# Researcher Assistant — Multi‑Agent AI System (Google & Kaggle 5‑Day Agents Intensive) 

A production‑style, multi‑agent research assistant that supports:
- **ArXiv search**
- **Remote A2A summarization**
- **Sequential multi‑agent orchestration**
- **Vector DB indexing using Pinecone - Optional**
- **Observability, metrics, logging**
- **Sessions & memory**
- **Evaluation**
- **Deployment as two cooperating FastAPI microservices**

---

## Table of contents
1. [Project overview](#project-overview)  
2. [Architecture](#architecture)  
3. [Agents & tools](#agents--tools)  
4. [A2A (remote agent) flow](#a2a-remote-agent-flow)  
5. [Observability & metrics](#observability--metrics)  
6. [Sessions & memory](#sessions--memory)  
7. [Evaluation harness](#evaluation-harness)  
8. [How to run (3 terminals)](#how-to-run-3-terminals)  
9. [API examples](#api-examples)  
10. [Environment & troubleshooting](#environment--troubleshooting)  
11. [Folder structure](#folder-structure)  
12. [Capstone requirements mapping](#capstone-requirements-mapping)  
13. [Notes for reviewers](#notes-for-reviewers)

---

## Project overview

**Researcher Assistant** automates literature search and review process for researchers.  
Given a user query (for example `"federated learning"`), the system:

1. Searches arXiv using a custom `arxiv_search` tool.  
2. Normalizes titles, abstracts, authors and pdf links.  
3. Assembles a combined prompt and sends it to a **remote summarizer** running as a separate service (A2A).  
4. Receives a strict JSON summary and list of top papers.  
5. Optionally upserts the results into Pinecone for vector retrieval. It is for experiemental and monitoring purposes.  
6. Tracks metrics, logs, and provides small admin endpoints for inspection.

This repo demonstrates the important agent concepts required by the Kaggle's Google Agents Intensive - Capstone Project: multi‑agent orchestration, tools, A2A, sessions & memory, observability, evaluation harness, and deployment.

---

## Architecture


![Project Flow Diagram](image.png)


Purpose of project files:
- `app.py`: coordinator + FastAPI API + root agent + arxiv search + Pinecone helpers + observability endpoints.
- `remote_agent.py`: remote summarizer A2A service (FastAPI).
- `models/model.py`: Pydantic request/response models.
- `evaluate.py`: evaluation harness driven by `integration.evalset.json`.
- `.env_sample`: Containes a sample variabe list of .env file
- `requirements.txt`: Containes the list of libraries to be installed to run the project.
- `logger.log`: Keeps log report for observability.
- `test_config.json`: Sets criteria for evaluation of functionalities
- `integration.evalset.json`: Sets the parameters and values to be used in evaluaion
- `eval_report.json`: Stores the output results generated from evaluation
- `agents/agent.py`: Triggers the web UI to use the agents involved in the project. When using adk web ui, you will see a option to select agents.
- `models/model.py`: Store all teh pydantic based validation schemas used in project
- `.gitignore`: List all the files and folders for git not to track to
---

## Agents & tools

### Agents
- **search_agent** (LlmAgent)
  - Uses `arxiv_search(query, max_results)` — an async tool that queries arXiv and returns normalized entries.
- **final_agent** (LlmAgent)
  - Uses `summarizer` wrapped as a `FunctionTool` that calls the remote A2A summarizer.
- **root_agent** (SequentialAgent)
  - Runs `search_agent` then `final_agent` in sequence to produce the final result.
- **remote_summarizer_agent** (running in `remote_agent.py`)
  - Forced to output a single valid JSON object with keys `summary` and `papers` (each paper has `title` and `url`).

### Tools
- `arxiv_search` — custom async fetcher (aiohttp + feedparser).
- `summarizer` — FunctionTool wrapper for `remote_summary_extract()` / A2A.
- Pinecone helpers:
  - `prepare_and_upsert(dense_index, records, namespace="arxiv")`
  - `retrieve_from_pinecone(query, top_k=5, namespace="arxiv")`


---

## A2A (remote agent) flow

- `app.py` calls `remote_summary_extract(query)` which posts to the remote A2A endpoint:
  ```
  POST http://localhost:9001/a2a
  payload: {"invocation_id": "<uuid>", "sender":"ResearchCoordinator", "query": "<string>"}
  ```
- `remote_agent.py` runs a `Runner` with `remote_summarizer_agent` that is instructed to return ONLY a single JSON object with `summary` and `papers`.
- `remote_agent.py` extracts JSON robustly (tries plain JSON parsing, then regex JSON block extraction).
- `app.py` receives the result, validates it, and uses it in the response and for upserts.

This separation demonstrates a real cross‑service agent integration (A2A) and is explicit proof of the A2A requirement.

---

## Observability & metrics

**Logging**
- Standard Python logging configured in `app.py` and `remote_agent.py`.
- ADK `LoggingPlugin()` is attached to runners for richer agent event logging.

**Metrics**
- `CountInvocationPlugin` counts agent runs and LLM requests, records last durations, and writes into `app.state.metrics`.
- `timed_async()` is used to record durations for important operations (arXiv search, summarizer call, upserts).

**Admin endpoints**
- `GET /admin/metrics` — returns a small JSON object with metrics such as:
  - `agent_count`, `llm_request_count`, `last_agent_run_ms`, `last_llm_request_ms`, `arxiv_search_ms`, `summarizer_ms`, `upsert_ms`.
- `GET /admin/last_agent_output` — returns the most recent agent output blob (summary, papers, raw_text, local debug blob).
- `GET /debug/watch_last_output?duration_sec=10` — pollable endpoint returning a timeline of observed last outputs (useful for debugging during long runs).

**ADK Web UI**
- Start with `adk web --port 8080` and inspect agent traces, tool calls, and event history at `http://localhost:8080`.

---

## Sessions & memory

- `InMemorySessionService` is used for session management (Runner is instantiated with `session_service=session_service`).
- `InMemoryMemoryService` is present and `memory_service.add_session_to_memory(session)` is called after agent runs to persist session state into the memory service.
- Notes: InMemory services are convenient for demo and evaluation. If you require persistence across restarts, swap to `DatabaseSessionService` (SQLite) or another backend — instructions for that are in **Environment & troubleshooting**.

---

## Evaluation harness

- `evaluate.py`:
  - Sets criteria for evaluation in `test_config`
  - Discovers the project module and expects symbols: `root_agent`, `Runner`, `session_service`, and `_extract_agent_text`.
  - Runs cases from `integration.evalset.json`.
  - Measures response similarity (token overlap), tool usage/trajectory, and special keyword hits.
  - Outputs `eval_report.json`.

**Important:** If you run evaluation, ensure the module exports `_extract_agent_text` (the repository contains robust extractors in `evaluate.py` and `remote_agent.py` that you can reuse).

---

## How to run (3 terminals)

**Prerequisites**
- Python 3.10+
- run libraries inside requirements.txt
- Google API key

**Terminal 1 — Start main orchestrator**
```bash
uvicorn app:app --reload --port 8000
```

**Terminal 2 — Start remote A2A summarizer**
```bash
uvicorn remote_agent:app --host 127.0.0.1 --port 9001 --reload
```

**Terminal 3 — ADK Web UI for traces**
```bash
adk web --port 8080 --log_level DEBUG
```

**Notes**
- After the services are up:
  - Use `POST /search` on `http://localhost:8000` to invoke the pipeline.
  - Inspect `http://localhost:8000/admin/metrics` and `http://localhost:8000/admin/last_agent_output` for observability.
  - Explore `http://localhost:8080` for ADK traces (if `adk web` is running).

---

## API examples

**Search request**
```
POST http://localhost:8000/search
Content-Type: application/json

{
  "query": "federated learning",
  "max_results": 10
}
```

**Metrics**
```
GET http://localhost:8000/admin/metrics
```

**Last agent output**
```
GET http://localhost:8000/admin/last_agent_output
```

**Watch timeline**
```
GET http://localhost:8000/debug/watch_last_output?duration_sec=10&poll_interval=0.5
```

---

## Environment & troubleshooting

**Example `.env` keys**
```
GOOGLE_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX=...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_CLOUD=...
PINECONE_MODEL=...
EMBEDDING_DIMENSION=1536
ARXIV_API=https://export.arxiv.org/api/query
REMOTE_SUMMARIZER_MODEL=gemini-2.5-flash-lite
```

**No Pinecone / local demo mode**
- If Pinecone credentials are not present, the app will attempt a defensive flow:
  - `prepare_and_upsert` will be skipped if `app.state.dense_index` is not present.
  - A local mirror (`app.state._local_index`) will still be used for retrieval examples.
- To run without Pinecone:
  - Omit Pinecone env vars and start the services. The pipeline will still run, but upserts will be disabled.

**Common issues**
- _Runner returns 429 RESOURCE_EXHAUSTED_: your LLM quota is exhausted or retries triggered. Reduce QPM or use a local small model.
- _remote A2A returns malformed JSON_: remote summarizer is required to return strict JSON; check `remote_agent` logs for parsing failures.
- _evaluate.py fails discovery_: ensure `_extract_agent_text` exists in the module or modify `evaluate.py` to point at a helper function included in the repo.

---

## Folder structure
```
.
├── app.py
├── remote_agent.py
├── models/
│   └── model.py
├── evaluate.py
├── integration.evalset.json
├── test_config.json
├── eval_report.json
├── README.md
├── agents/
│    └── agent.py
├── .env_sample
├── .gitignore
├── requirements.txt
```

---

## Capstone requirements mapping

| Requirement | Demonstrated in repo |
|-------------|----------------------|
| Multi-agent system | SequentialAgent (`root_agent`) |
| Tools (custom / FunctionTool) | `arxiv_search`, `summarizer` FunctionTool |
| A2A Protocol | `remote_agent.py` / `remote_summary_extract()` |
| Sessions & Memory | `InMemorySessionService` + `InMemoryMemoryService` |
| Observability | `CountInvocationPlugin`, `/admin/metrics`, `adk web` |
| Agent evaluation | `evaluate.py` + `integration.evalset.json` |
| Deployment | `app.py` and `remote_agent.py` FastAPI services |

---

## Notes for reviewers

- **Ports**: `app.py` (8000), `remote_agent.py` (9001), `adk web` (8080).
- **Evaluator**: `evaluate.py` is provided to replicate structured tests. It expects an `_extract_agent_text` helper — the repo includes extractors in `app.py` and `remote_agent.py`.
- **Persistence tradeoff**: InMemory session/memory is chosen for demonstartion in Capstone Project only.
- **Reproducibility**: If reviewers have no Pinecone account, tests still run if you skip the upsert step — provide empty/missing Pinecone keys.

---

## License & acknowledgements

This project was built for the Kaggle 5‑Day Agents Capstone. Portions of code patterns follow the ADK examples and Kaggle notebooks provided during the course.


