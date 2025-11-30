from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.model import SearchRequest, SearchResponse, PaperItem
import aiohttp
import os
import uuid
import json
import asyncio
from typing import List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager
import feedparser
import uvicorn
import logging
import time
# ADK / Google GenAI libraries
from google.genai import types
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.function_tool import FunctionTool
from google.adk.memory import InMemoryMemoryService
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

# ---------------- Logging ----------------
logging.basicConfig(
    filename="logger.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("researcher_assistant")

# ---------------- Environment / setup ----------------
load_dotenv()

# In-memory session and memory services (suitable for demo / prototype)
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()
preload_tool = PreloadMemoryTool()

logger.info("preload_tool: %s", [n for n in dir(preload_tool) if not n.startswith("_")])
logger.info("memory_service: %s", [n for n in dir(memory_service) if not n.startswith("_")])

# Config from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ARXIV_API = os.getenv("ARXIV_API")

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment. Add to .env")

app = FastAPI(title="Researcher Assistant", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Retry options for LLM HTTP calls
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# ---------------- Plugin: basic invocation / timing metrics ----------------
class CountInvocationPlugin(BasePlugin):
    """
    Simple plugin to count agent runs and LLM requests and record simple timing metrics
    into app.state.metrics for observability.
    """
    def __init__(self) -> None:
        super().__init__(name="count_invocation")
        self.agent_count = 0
        self.llm_request_count = 0
        self.last_agent_run_ms = None
        self.last_llm_request_ms = None

    def _ensure_metrics_slot(self):
        app.state.metrics = getattr(app.state, "metrics", {}) or {}
        return app.state.metrics

    async def before_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> None:
        callback_context._agent_start = time.perf_counter()
        self.agent_count += 1
        logging.info(f"[CountInvocationPlugin] Agent run #{self.agent_count} for agent={getattr(agent, 'name', None)}")
        try:
            m = self._ensure_metrics_slot()
            m["agent_count"] = self.agent_count
            app.state.metrics = m
        except Exception:
            logging.exception("failed to write agent_count to app.state.metrics")

    async def after_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext, result: Any = None) -> None:
        try:
            elapsed = max(0.0, (time.perf_counter() - getattr(callback_context, "_agent_start", time.perf_counter())) * 1000.0)
            self.last_agent_run_ms = elapsed
            logging.info(f"[CountInvocationPlugin] Agent {getattr(agent, 'name', None)} finished in {elapsed:.1f} ms")
            m = self._ensure_metrics_slot()
            m["last_agent_run_ms"] = elapsed
            m["agent_count"] = self.agent_count
            app.state.metrics = m
        except Exception:
            logging.exception("after_agent_callback error")

    async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        callback_context._llm_start = time.perf_counter()
        self.llm_request_count += 1
        prompt_len = len(getattr(llm_request, "prompt", "") or "")
        logging.info(f"[CountInvocationPlugin] LLM request #{self.llm_request_count}; prompt_len={prompt_len}")
        try:
            m = self._ensure_metrics_slot()
            m["llm_request_count"] = self.llm_request_count
            app.state.metrics = m
        except Exception:
            logging.exception("failed to write llm_request_count to app.state.metrics")

    async def after_model_callback(self, *, callback_context: CallbackContext, llm_response: Any = None) -> None:
        try:
            elapsed = max(0.0, (time.perf_counter() - getattr(callback_context, "_llm_start", time.perf_counter())) * 1000.0)
            self.last_llm_request_ms = elapsed
            logging.info(f"[CountInvocationPlugin] LLM response received in {elapsed:.1f} ms")
            m = self._ensure_metrics_slot()
            m["last_llm_request_ms"] = elapsed
            m["llm_request_count"] = self.llm_request_count
            app.state.metrics = m
        except Exception:
            logging.exception("after_model_callback error")

# ---------------- arXiv helpers ----------------
def _normalize_feed_entry(e) -> dict:
    """
    Normalize a feedparser entry into a consistent paper dict.
    """
    arxiv_id = (e.get("id") or "").rsplit("/", 1)[-1] if e.get("id") else None
    authors = [a.get("name") for a in e.get("authors", []) if a.get("name")]
    pdf_url = None
    for link in e.get("links", []):
        if link.get("type") == "application/pdf" or (link.get("title") and "pdf" in link.get("title", "").lower()):
            pdf_url = link.get("href")
            break

    published_iso = None
    if getattr(e, "published_parsed", None):
        try:
            published_iso = datetime(*e.published_parsed[:6]).isoformat()
        except Exception:
            published_iso = None

    return {
        "id": arxiv_id or e.get("id") or str(uuid.uuid4()),
        "title": (e.get("title") or "").replace("\n", " ").strip(),
        "authors": authors,
        "abstract": (e.get("summary") or "").strip(),
        "link": e.get("link"),
        "date": published_iso,
        "pdf_url": pdf_url,
        "raw_entry_id": e.get("id"),
    }

async def arxiv_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Query the arXiv API and return a list of normalized paper dicts.
    Uses feedparser in a thread to avoid blocking the event loop.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance"
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(ARXIV_API, params=params) as resp:
            resp.raise_for_status()
            xml_text = await resp.text()
            feed = await asyncio.to_thread(feedparser.parse, xml_text)
            out = []
            for entry in getattr(feed, "entries", []):
                try:
                    out.append(_normalize_feed_entry(entry))
                except Exception:
                    # Skip entries that fail to parse
                    continue
            return out

# ---------------- Generic helpers ----------------
async def timed_async(fn, *args, metric_key: str = None, **kwargs):
    """
    Run an async function and measure elapsed time in milliseconds.
    Optionally store the duration in app.state.metrics under metric_key.
    """
    start = time.perf_counter()
    res = await fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logging.info(f"[timed_async] {getattr(fn, '__name__', str(fn))} took {elapsed_ms:.1f} ms")
    try:
        app.state.metrics = getattr(app.state, "metrics", {})
        if metric_key:
            app.state.metrics[metric_key] = elapsed_ms
    except Exception:
        # Non-critical: metrics failure should not block response
        pass
    return res

# ---------------- Remote summarizer integration ----------------
async def remote_summary_extract(query: str = "", docs: Optional[List[dict]] = None,
                                 remote_url: Optional[str] = None, timeout_sec: int = 25) -> dict:
    """
    Call the remote summarizer service (A2A endpoint) and validate/normalize its response.

    Returns:
      - On success: {"summary": "<text>", "papers": [<paper dicts>] }
      - On failure: {"error": "<code>", "raw": "<diagnostic info>"}
    """
    url = remote_url or "http://localhost:9001/a2a"
    payload = {
        "invocation_id": str(uuid.uuid4()),
        "sender": "ResearchCoordinator",
        "query": query or "",
        "docs": docs or []
    }
    start = time.perf_counter()
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception as e:
        logging.exception("a2a call failed")
        try:
            app.state.metrics = getattr(app.state, "metrics", {})
            app.state.metrics["a2a_summarizer_ms"] = (time.perf_counter() - start) * 1000.0
            app.state.metrics["a2a_summarizer_fail_count"] = app.state.metrics.get("a2a_summarizer_fail_count", 0) + 1
        except Exception:
            pass
        return {"error": "remote_call_failed", "raw": str(e)}

    # record timing and increment counter
    try:
        app.state.metrics = getattr(app.state, "metrics", {})
        app.state.metrics["a2a_summarizer_ms"] = (time.perf_counter() - start) * 1000.0
        app.state.metrics["a2a_summarizer_count"] = app.state.metrics.get("a2a_summarizer_count", 0) + 1
    except Exception:
        pass

    # Validate response shape
    if not isinstance(data, dict):
        return {"error": "remote_nonjson_response", "raw": str(data)[:4000]}
    if not data.get("ok"):
        return {"error": "remote_ok_false", "raw": data}

    result = data.get("result")

    # If result is string, try to parse JSON out of it
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return {"error": "remote_result_not_json", "raw": result[:4000]}

    if not isinstance(result, dict):
        return {"error": "remote_result_bad_shape", "raw": str(result)[:4000]}

    if result.get("error"):
        return {"error": result.get("error"), "raw": result.get("raw") or ""}

    # If remote returned a 'papers' list, normalize handling
    papers = result.get("papers") or []
    if isinstance(papers, list):
        if papers and isinstance(papers[0], dict) and (
            papers[0].get("id") or papers[0].get("abstract") or papers[0].get("pdf_url") or papers[0].get("arxiv_url")
        ):
            # Looks like canonical paper objects; preserve as-is
            return {"summary": (result.get("summary") or "").strip(), "papers": papers}
        # Legacy shape: list of {title,url}
        normalized = []
        for p in papers:
            if not isinstance(p, dict):
                continue
            title = (p.get("title") or "").strip()
            url = (p.get("url") or p.get("pdf_url") or p.get("link") or "").strip()
            normalized.append({"title": title, "url": url})
        return {"summary": (result.get("summary") or "").strip(), "papers": normalized}

    # If the remote returned only a summary string, attach the original docs
    summary_txt = (result.get("summary") or "").strip() if isinstance(result.get("summary"), str) else ""
    if summary_txt:
        return {"summary": summary_txt, "papers": docs or []}

    return {"error": "remote_result_missing_fields", "raw": result}

async def summarizer_tool(query: str = "", docs: Optional[List[dict]] = None,
                          remote_url: Optional[str] = None, timeout_sec: int = 25):
    """
    Wrapper used as a FunctionTool for agents. It will:
      - attempt to auto-fill docs from app.state if not provided
      - call remote_summary_extract and return its structured result
    """
    try:
        logger.info("[summarizer_tool] called; query_len=%d docs_len=%s", len((query or "")), None if docs is None else len(docs))
    except Exception:
        pass

    # Auto-fill canonical docs from app.state if caller omitted docs
    if not docs:
        try:
            last = getattr(app.state, "last_agent_output", None) or {}
            cand = last.get("papers") or getattr(app.state, "arxiv_candidates", None) or []
            if isinstance(cand, list) and cand:
                docs = cand
                logger.info("[summarizer_tool] auto-filled docs from app.state (len=%d)", len(docs))
        except Exception:
            logger.exception("failed to auto-fill docs; proceeding without docs")

    res = await remote_summary_extract(query=query, docs=docs, remote_url=remote_url, timeout_sec=timeout_sec)

    # Log some basic signals about returned papers
    try:
        papers = res.get("papers") if isinstance(res, dict) else None
        logger.info("[summarizer_tool] summarizer returned summary_len=%d papers_len=%s",
                    len((res.get("summary") or "")), None if papers is None else len(papers))
        if papers:
            first = papers[0] if isinstance(papers, list) and papers else {}
            has_link = bool(first.get("pdf_url") or first.get("arxiv_url") or first.get("link") or first.get("url"))
            logger.info("[summarizer_tool] first_paper_has_link=%s", has_link)
    except Exception:
        pass

    return res

# ---------------- Display / formatting ----------------
def build_display_text(summary: Optional[str], papers: List[dict]) -> str:
    """
    Build a simple human-readable block the UI can render verbatim.
    """
    lines = []
    if summary:
        lines.append("Summary:")
        lines.append(summary.strip())
        lines.append("")

    if papers:
        for p in papers:
            title = (p.get("title") or p.get("paper_title") or p.get("id") or "(no title)").strip()
            url = (p.get("pdf_url") or p.get("link") or p.get("url") or p.get("arxiv_url") or "").strip()
            if url:
                lines.append(f"Title: {title}  Link: {url}")
            else:
                lines.append(f"Title: {title}")
    else:
        lines.append("No papers found.")

    return "\n".join(lines)

# ---------------- Agent definitions ----------------
search_agent = LlmAgent(
    name="arxiv_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, api_key=GOOGLE_API_KEY),
    instruction="""
        Call the provided tool `arxiv_search` with the user's query and RETURN ONLY the tool/function RESPONSE OBJECT.
        The structured tool response will be passed to the next agent.
    """,
    tools=[FunctionTool(func=arxiv_search), preload_tool],
    description="Search arXiv and write structured results to state",
    output_key="arxiv_candidates",
)

final_agent = LlmAgent(
    name="final_summarizer_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, api_key=GOOGLE_API_KEY),
    instruction="""
        1) Call the FunctionTool "summarizer" with:
           - query: 1-3 line combined context built from titles + abstracts (or the user query).
           - docs: canonical paper objects (id, title, abstract, pdf_url/arxiv_url when available).
        2) After the tool returns, produce ONLY a plain-text message formatted like:

Summary:
<one to three sentence concise summary>

Here are the papers with links:
1. <Title 1> — <url1>
2. <Title 2> — <url2>
...
        3) If the tool returns no summary, synthesize a short human summary from the first 3 docs or from the user query.
    """,
    tools=[FunctionTool(func=summarizer_tool)],
)

root_agent = SequentialAgent(
    name="ResearchCoordinator",
    sub_agents=[search_agent, final_agent],
    description="Run search_agent followed by final_summarizer_agent",
)

# ---------------- Lifespan / startup ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize simple metrics and field map used by the app
    app.state.metrics = getattr(app.state, "metrics", {}) or {}
    app.state.field_map = getattr(app.state, "field_map", {}) or {"text": "text"}
    yield

app.router.lifespan_context = lifespan

# ---------------- HTTP routes ----------------
@app.get("/")
def root():
    return {"ok": True, "service": "ResearcherAssistant", "note": "POST /search"}

@app.post("/search", response_model=SearchResponse)
async def search_papers(req: SearchRequest):
    """
    Main search endpoint. High-level flow:
      1. Validate request and initialize UI blob (app.state.last_agent_output).
      2. Run arXiv search (synchronous to ensure canonical docs available).
      3. Start the ADK Runner in the background to stream detailed agent events.
      4. Immediately call the remote summarizer (server-side) with canonical docs to provide a fast
         user-facing summary. This runs in parallel with the background runner.
      5. Normalize returned papers, publish last_agent_output, and return SearchResponse using canonical docs.
    """
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty query")
    maxr = min(req.max_results or 5, 10, 50)

    # Initialize UI-visible output blob
    app.state.last_agent_output = getattr(app.state, "last_agent_output", {}) or {}
    app.state.last_agent_output.update({
        "raw_text": "",
        "summary": None,
        "papers": [],
        "links": [],
        "display_text": None,
        "agent_final_text": "",
    })

    logging.info("HYBRID SEARCH q=%s maxr=%d", q, maxr)

    try:
        # Attach metrics/tracking plugin instance for this request
        count_invocation_plugin = CountInvocationPlugin()
        app.state.count_invocation_plugin = count_invocation_plugin

        # Fetch canonical arXiv candidates (used by summarizer and stored for UI)
        arxiv_candidates = await timed_async(arxiv_search, q, max_results=min(maxr, 5), metric_key="arxiv_search_ms")
        app.state.arxiv_candidates = arxiv_candidates or []

        # Create a Runner for the root_agent. This will run in the background and publish event stream.
        runner = Runner(
            agent=root_agent,
            app_name="agents",
            session_service=session_service,
            memory_service=memory_service,
            plugins=[LoggingPlugin(), app.state.count_invocation_plugin],
        )

        async def _run_and_persist_session(runner: Runner, session_obj, user_text: str):
            """
            Background task: run the agent runner asynchronously and update app.state.last_agent_output
            with event snapshots and final text when available.
            """
            final_response_text = None
            events_seen = []
            try:
                content = types.Content(role="user", parts=[types.Part(text=user_text)])

                async for event in runner.run_async(
                    user_id=session_obj.user_id,
                    session_id=session_obj.id,
                    new_message=content,
                ):
                    # Build compact event representation for logs and UI
                    try:
                        ev_repr = {
                            "id": getattr(event, "id", None),
                            "author": getattr(event, "author", None),
                            "type": getattr(event, "type", None),
                        }
                        if getattr(event, "content", None) and getattr(event.content, "parts", None):
                            ev_repr["text_snippet"] = (event.content.parts[0].text or "")[:500]
                        events_seen.append(ev_repr)
                        logger.info("runner event: %s", ev_repr)
                    except Exception:
                        logger.exception("error serializing event")

                    # Publish incremental snapshot to app.state.last_agent_output
                    try:
                        cur = getattr(app.state, "last_agent_output", {}) or {}
                        cur.update({
                            "agent_events": events_seen[-50:],
                            "agent_session_id": getattr(session_obj, "id", None),
                            "agent_last_event_time": datetime.utcnow().isoformat() + "Z",
                        })
                        app.state.last_agent_output = cur
                    except Exception:
                        logger.exception("failed to publish incremental last_agent_output")

                    # Detect final-response signals and publish final text immediately
                    try:
                        if getattr(event, "is_final_response", None) and callable(event.is_final_response) and event.is_final_response():
                            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                                final_response_text = event.content.parts[0].text

                            try:
                                cur = getattr(app.state, "last_agent_output", {}) or {}
                                cur.update({
                                    "agent_final_text": final_response_text,
                                    "agent_final_published_at": datetime.utcnow().isoformat() + "Z",
                                    "agent_events": events_seen[-50:],
                                })
                                app.state.last_agent_output = cur
                            except Exception:
                                logger.exception("failed to publish agent_final_text (is_final_response)")
                            break

                        if getattr(event, "final", False):
                            final_response_text = getattr(event.content.parts[0], "text", final_response_text) or final_response_text

                            try:
                                cur = getattr(app.state, "last_agent_output", {}) or {}
                                cur.update({
                                    "agent_final_text": final_response_text,
                                    "agent_final_published_at": datetime.utcnow().isoformat() + "Z",
                                    "agent_events": events_seen[-50:],
                                })
                                app.state.last_agent_output = cur
                            except Exception:
                                logger.exception("failed to publish agent_final_text (final)")
                            break
                    except Exception:
                        logger.exception("error checking final response")

            except Exception:
                logger.exception("_run_and_persist_session: runner error")
            finally:
                # On completion, merge events and final text into the UI blob without overwriting summary/papers
                try:
                    cur = getattr(app.state, "last_agent_output", {}) or {}
                    fallback_text = cur.get("agent_final_text") or cur.get("summary") or cur.get("display_text") or cur.get("human_readable") or ""
                    cur.update({
                        "agent_events": events_seen[-20:],
                        "agent_final_text": (final_response_text or fallback_text),
                        "agent_session_id": getattr(session_obj, "id", None),
                    })

                    # Ensure UI-facing fields are present and reasonably sized
                    try:
                        final_text_choice = cur.get("agent_final_text") or ""
                        if final_text_choice and len(final_text_choice) > 8000:
                            final_text_choice = final_text_choice[:8000] + "…"

                        cur.setdefault("agent_final_text", final_text_choice)
                        cur.setdefault("agent_session_id", cur.get("agent_session_id") or getattr(session_obj, "id", None))
                        cur.setdefault("published_at", cur.get("agent_final_published_at") or datetime.utcnow().isoformat() + "Z")

                        logger.info("DEBUG publish snapshot: agent_final_text_present=%s session_id=%s summary_len=%d",
                                    bool(final_text_choice), cur.get("agent_session_id"), len((cur.get("summary") or "")))
                    except Exception:
                        logger.exception("failed to ensure agent_final_text/session_id on last_agent_output")

                    app.state.last_agent_output = cur
                    logger.info(
                        "attached runner events/final_text to last_agent_output (events=%d, final_present=%s)",
                        len(events_seen), bool(final_response_text or fallback_text)
                    )
                except Exception:
                    logger.exception("failed to attach runner events to last_agent_output")

            return final_response_text

        # Create a session and start the runner in background so UI can stream events while we summarise immediately
        session = await session_service.create_session(app_name=runner.app_name or "agents", user_id=str(uuid.uuid4()))
        asyncio.create_task(_run_and_persist_session(runner, session, f"User query: {q}\n\nPlease follow the instructions carefully"))
    except Exception:
        logging.exception("Failed to start runner (non-fatal)")

    # Build combined text for summarizer from canonical arXiv candidates
    parts = []
    for i, c in enumerate((arxiv_candidates or [])[:8], start=1):
        title = (c.get("title") or "").strip()
        abstract = (c.get("abstract") or c.get("summary") or "").strip()
        url = (c.get("pdf_url") or c.get("link") or "").strip()
        if title:
            if url:
                parts.append(f"{i}. {title} ({url})\n{abstract}\n")
            else:
                parts.append(f"{i}. {title}\n{abstract}\n")
    combined = "Summarize these papers in 1-3 sentences and return JSON with keys 'summary' and 'papers' (title+url):\n\n"
    combined += ("\n".join(parts) if parts else q)

    # Perform immediate (server-side) call to remote summarizer so UI receives a fast summary.
    start = time.perf_counter()
    summarizer_result = await remote_summary_extract(query=combined, docs=arxiv_candidates, remote_url=None, timeout_sec=30)
    app.state.metrics = getattr(app.state, "metrics", {}) or {}
    app.state.metrics["summarizer_ms"] = (time.perf_counter() - start) * 1000.0

    # --- Normalize returned papers into UI-friendly shapes ---
    raw_papers = summarizer_result.get("papers") or []
    normalized_papers = []
    links_for_ui = []

    def _pick_url(p):
        return (p.get("pdf_url") or p.get("arxiv_url") or p.get("link") or p.get("url") or "").strip()

    for p in raw_papers:
        if not isinstance(p, dict):
            continue
        title = (p.get("title") or p.get("paper_title") or p.get("id") or "")[:1000]
        url = _pick_url(p)
        normalized = {
            "id": p.get("id") or p.get("raw_entry_id") or str(uuid.uuid4()),
            "title": title,
            "abstract": p.get("abstract") or p.get("summary") or "",
            "pdf_url": p.get("pdf_url"),
            "arxiv_url": p.get("arxiv_url") or p.get("link") or None,
            "link": url or None,
            "raw": p,
        }
        normalized_papers.append(normalized)
        if url:
            links_for_ui.append({"title": title, "url": url})

    # Fallback: if remote returned no papers, fall back to canonical arXiv candidates
    if not normalized_papers and (arxiv_candidates or []):
        for c in (arxiv_candidates or [])[:10]:
            title = (c.get("title") or "")[:1000]
            url = (c.get("pdf_url") or c.get("link") or "")
            normalized_papers.append({
                "id": c.get("id") or str(uuid.uuid4()),
                "title": title,
                "abstract": c.get("abstract") or "",
                "pdf_url": c.get("pdf_url"),
                "arxiv_url": c.get("link"),
                "link": url or None,
                "raw": c,
            })
            if url:
                links_for_ui.append({"title": title, "url": url})

    # Prepare raw_text for debugging/UI
    try:
        raw_text = json.dumps(summarizer_result, ensure_ascii=False)
    except Exception:
        raw_text = json.dumps({"error": "raw_text_serialization_failed", "repr": str(summarizer_result)[:2000]}, ensure_ascii=False)

    # Build human-readable block for UI rendering
    try:
        human_readable = build_display_text(
            summary=summarizer_result.get("summary"),
            papers=normalized_papers or (raw_papers or [])
        )
    except Exception:
        human_readable = None
        try:
            if summarizer_result.get("summary"):
                human_readable = "Summary:\n\n" + summarizer_result.get("summary") + "\n\n"
                if normalized_papers:
                    human_readable += "Top papers:\n"
                    for i, p in enumerate(normalized_papers[:5], start=1):
                        display_url = p.get("link") or p.get("pdf_url") or p.get("arxiv_url") or ""
                        human_readable += f"{i}. {p.get('title','(no title)')} — {display_url}\n"
        except Exception:
            human_readable = None

    # Merge and publish last_agent_output (do not overwrite existing summary/papers if set)
    try:
        cur = getattr(app.state, "last_agent_output", {}) or {}
        cur.update({
            "raw_text": raw_text,
            "summary": summarizer_result.get("summary"),
            "papers": raw_papers,
            "normalized_papers": normalized_papers,
            "links": links_for_ui,
            "display_text": human_readable,
            "human_readable": human_readable,
            "source": "hybrid-orchestrator",
        })
        app.state.last_agent_output = cur

        logging.info("Published last_agent_output (summary_present=%s, papers=%d, normalized_papers=%d, links=%d)",
                     bool(cur.get("summary")), len(raw_papers), len(normalized_papers), len(links_for_ui))
    except Exception:
        logger.exception("Failed to publish last_agent_output (non-fatal)")

    # Build response using canonical arXiv candidates (these are the authoritative paper objects returned on /search)
    papers_items = []
    for e in (arxiv_candidates or [])[:maxr]:
        papers_items.append(PaperItem(
            id=e.get("id") or str(uuid.uuid4()),
            title=e.get("title") or "",
            abstract=e.get("abstract") or "",
            authors=e.get("authors") or [],
            pdf_url=e.get("pdf_url") or None,
            published=None,
            arxiv_url=e.get("link")
        ))
    return SearchResponse(
        query=q,
        count=len(papers_items),
        papers=papers_items,
        summary=app.state.last_agent_output.get("summary"),
        top_papers=[p.dict() for p in papers_items[:5]],
        top_paper_links=[{"title": p.title, "url": p.pdf_url or p.arxiv_url or ""} for p in papers_items[:5]],
        display_text=app.state.last_agent_output.get("display_text"),
        human_readable=app.state.last_agent_output.get("human_readable"),
    )

# ---------------- Admin / observability endpoints ----------------
@app.get("/admin/metrics")
def get_metrics():
    """
    Return a small metrics snapshot for basic observability.
    """
    try:
        metrics = getattr(app.state, "metrics", {}) or {}
        resp = {
            "agent_count": int(metrics.get("agent_count") or 0),
            "llm_request_count": int(metrics.get("llm_request_count") or 0),
            "arxiv_search_ms": float(metrics.get("arxiv_search_ms") or 0.0),
            "summarizer_ms": float(metrics.get("summarizer_ms") or 0.0)
        }
        return {"ok": True, "metrics": resp}
    except Exception as e:
        logging.exception("Error reading metrics")
        return {"ok": False, "error": str(e)}

@app.get("/admin/last_agent_output")
def last_agent_output():
    """
    Return a compact UI-friendly blob:
      - summary, display_text/human_readable
      - papers (raw list returned by summarizer)
      - links (simple {title,url} list)
      - raw_text (raw string from summarizer) and raw_blob for debugging
    """
    out = getattr(app.state, "last_agent_output", None) or {}
    return {
        "ok": True,
        "summary": out.get("summary"),
        "display_text": out.get("display_text"),
        "human_readable": out.get("human_readable"),
        "papers": out.get("papers"),
        "links": out.get("links"),
        "raw_text": out.get("raw_text"),
        "raw_blob": out,
    }

@app.get("/debug/watch_last_output")
def debug_watch_last_output(duration_sec: int = 10, poll_interval: float = 0.5):
    """
    Poll the application state for changes to last_agent_output and return a timeline.
    Useful for debugging and UI polling during demonstrations.
    """
    start = time.time()
    timeline = []
    seen = None
    while time.time() - start < duration_sec:
        try:
            s = getattr(app, "state", None)
            cur = getattr(s, "last_agent_output", None)
            metrics = getattr(s, "metrics", None)
            if cur is None:
                summary = None
            else:
                try:
                    summary = json.dumps({"summary": cur.get("summary"), "links_len": len(cur.get("links") or [])}, ensure_ascii=False)[:500]
                except Exception:
                    try:
                        summary = str(cur)[:500]
                    except Exception:
                        summary = "<unserializable>"
            if summary != seen:
                timeline.append({"t": time.time(), "summary": summary, "metrics": metrics})
                seen = summary
        except Exception as e:
            timeline.append({"t": time.time(), "error": str(e)})
        time.sleep(poll_interval)
    return {"ok": True, "timeline": timeline}

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)