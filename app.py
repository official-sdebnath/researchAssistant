from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from models.model import SearchRequest, SearchResponse, PaperItem
import aiohttp
from pinecone import Pinecone
import os 
import uuid 
import json 
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime   
from contextlib import asynccontextmanager
import feedparser
from datetime import date
import uvicorn
import logging
import time
# Google adk based libraries
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

# Logger Invoked

logging.basicConfig(
    filename="logger.log",
    level=logging.INFO,  
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("researcher_assistant")

# ------------------ Initial Setup ------------------
load_dotenv()

# In-memory session service for ADK agents
session_service = InMemorySessionService()

# In-memory memory service for ADK agents
memory_service = InMemoryMemoryService()

preload_tool = PreloadMemoryTool()

logger.info("preload_tool: %s", [n for n in dir(preload_tool) if not n.startswith("_")])
logger.info("memory_service: %s", [n for n in dir(memory_service) if not n.startswith("_")])

# ------------------------ Config ------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_MODEL = os.getenv("PINECONE_MODEL")
APP_EMBED_DIM = int(os.getenv("EMBEDDING_DIMENSION"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ARXIV_API = os.getenv("ARXIV_API")

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment. Add to .env")

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment. Add to .env")

app = FastAPI(
    title="Researcher Assistant",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ---------------- Plugins ---------------------

class CountInvocationPlugin(BasePlugin):
    """
    Counts agent runs and LLM requests, records timings,
    and writes simple metrics to app.state.metrics (app-scoped).
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
            logging.exception("[CountInvocationPlugin] failed to write agent_count to app.state.metrics")

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
            logging.exception("[CountInvocationPlugin] after_agent_callback error")

    async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        callback_context._llm_start = time.perf_counter()
        self.llm_request_count += 1
        logging.info(f"[CountInvocationPlugin] LLM request #{self.llm_request_count}; prompt_len={len(getattr(llm_request, 'prompt', '') or '')}")
        try:
            m = self._ensure_metrics_slot()
            m["llm_request_count"] = self.llm_request_count
            app.state.metrics = m
        except Exception:
            logging.exception("[CountInvocationPlugin] failed to write llm_request_count to app.state.metrics")

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
            logging.exception("[CountInvocationPlugin] after_model_callback error")

# ---------------- arXiv Helpers ---------------------

def _normalize_feed_entry(e) -> dict:
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

async def arxiv_search(query: str, max_results: int = 10) -> List[dict]:
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
            # feedparser.parse is synchronous, run it in a thread
            feed = await asyncio.to_thread(feedparser.parse, xml_text)
            out = []
            for entry in getattr(feed, "entries", []):
                try:
                    out.append(_normalize_feed_entry(entry))
                except Exception:
                    # swallow per-entry parse problems
                    continue
            return out

# ---------------- Pinecone Helpers ---------------------

def sanitize_metadata_for_pinecone(meta: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
            continue
        if isinstance(v, list):
            clean[k] = [str(x) for x in v if x is not None]
            continue
        if isinstance(v, (datetime, date)):
            clean[k] = v.isoformat()
            continue
        try:
            clean[k] = json.dumps(v, ensure_ascii=False)
        except Exception:
            clean[k] = str(v)
    return clean

def prepare_and_upsert(dense_index, records: List[dict], namespace: str = "default"):
    """
    Upserts records into Pinecone. Builds records like:
      {"_id": "...", "<text_field>": "...", <meta fields> }

    This version DOES NOT mirror into a local in-process index.
    It tries to call common SDK methods in a defensive manner:
      - upsert_records (text-based, integrated-embedding)
      - upsert (vector-based)
    """
    if not records:
        return {"ok": True, "note": "no records"}

    # Determine the text field name expected by the index 
    field_map = getattr(app.state, "field_map", {}) or {}
    text_field = field_map.get("text", "text")

    records_payload = []
    for rec in records:
        rid = rec.get("id") or rec.get("_id") or str(uuid.uuid4())
        meta = sanitize_metadata_for_pinecone(rec.get("metadata", {}) or {})
        text = (rec.get("text") or rec.get("chunk_text") or "")[:3000]

        # Build the record with the correct text field name
        r = {"_id": str(rid), text_field: str(text)}
        # Merge metadata as top-level keys (avoid collisions)
        for k, v in meta.items():
            if k in ("_id", text_field):
                r[f"meta_{k}"] = v
            else:
                r[k] = v
        records_payload.append(r)

    try:
        if hasattr(dense_index, "upsert_records"):
            return dense_index.upsert_records(namespace, records_payload) or {"ok": True, "count": len(records_payload)}
        if hasattr(dense_index, "upsert"):
            raise RuntimeError("Index supports 'upsert' (vector upsert) but no vectors provided. Use integrated-embedding index or supply vectors.")
        if callable(getattr(dense_index, "upsert_records", None)):
            return dense_index.upsert_records(namespace, records_payload) or {"ok": True, "count": len(records_payload)}
        return {"ok": False, "error": "no_supported_upsert_method_on_index"}
    except Exception as e:
        logging.exception("Pinecone upsert failed")
        return {"ok": False, "error": str(e)}

def retrieve_from_pinecone(query: str, top_k: int = 5, namespace: str = "arxiv") -> List[dict]:
    """
    Query the Pinecone index using text or vector search and return list of:
      { "id": <id>, "text": <picked_text>, "metadata": {...} }

    This function is defensive across SDK versions:
     - Prefers `search_records` / `search` / `query`-style methods that accept query text or vector
     - Falls back to returning [] if index is not available
    """
    if not query:
        return []

    idx = getattr(app.state, "dense_index", None)
    if idx is None:
        return []

    # Decide which method to call depending on the SDK
    try:
        # 1) If index exposes search_records (text/record search), use it
        if hasattr(idx, "search_records"):
            # Many Pinecone SDKs accept: search_records(query=..., top_k=..., namespace=...)
            resp = idx.search_records(query=query, top_k=top_k, namespace=namespace)
            hits = getattr(resp, "matches", None) or resp.get("matches", []) if isinstance(resp, dict) else resp
            results = []
            for h in hits[:top_k]:
                # adapt to match object shapes
                metadata = getattr(h, "metadata", None) or h.get("metadata", {}) if isinstance(h, dict) else {}
                text = (
                    getattr(h, "record", None) and getattr(h.record, "text", None)
                ) or h.get("record", {}).get("text") if isinstance(h, dict) else None
                # fallback to metadata title or other fields
                if not text:
                    text = metadata.get("title") or metadata.get("abstract") or ""
                results.append({"id": getattr(h, "id", None) or h.get("id"), "text": text, "metadata": dict(metadata)})
            return results

        # 2) If index has query or search method accepting vectors/text, call query/search
        if hasattr(idx, "query") or hasattr(idx, "search"):
            # Some SDKs expose `query` which expects a vector OR a record id.
            # Newer docs also provide `search_records` for text-based search; we've tried that above.
            # As we don't have an embedding client here, try a text-based search call where supported.
            if hasattr(idx, "search"):
                resp = idx.search(query=query, top_k=top_k, namespace=namespace)
            else:
                resp = idx.query(query=query, top_k=top_k, namespace=namespace)
            # resp shape varies: adapt by looking for 'matches' or 'results'
            matches = getattr(resp, "matches", None) or resp.get("matches", []) if isinstance(resp, dict) else resp
            out = []
            for m in matches[:top_k]:
                meta = getattr(m, "metadata", None) or m.get("metadata", {}) if isinstance(m, dict) else {}
                text = meta.get("title") or meta.get("abstract") or (m.get("text") if isinstance(m, dict) else "")
                out.append({"id": getattr(m, "id", None) or m.get("id"), "text": text, "metadata": dict(meta)})
            return out

        # 3) No supported search API found
        return []
    except Exception:
        logging.exception("retrieve_from_pinecone failed")
        return []
# ------------------ Generic Helpers ------------------

def merge_with_arxiv_candidates(result: dict, arxiv_candidates: List[dict]) -> dict:
    """
    Best-effort merge: if remote result papers lack urls, try to match by title to arxiv_candidates and attach pdf_url/link.
    """
    try:
        if not isinstance(result, dict):
            return result
        if not arxiv_candidates:
            return result
        # build title->url map (lowercase normalized)
        title_map = {}
        for c in arxiv_candidates:
            t = (c.get("title") or "").strip()
            u = (c.get("pdf_url") or c.get("link") or "").strip()
            if t:
                title_map[t.lower()] = u
        papers = result.get("papers") or []
        merged = []
        for p in papers:
            if not isinstance(p, dict):
                continue
            t = (p.get("title") or "").strip()
            u = (p.get("url") or p.get("link") or "").strip()
            if not u and t and title_map.get(t.lower()):
                u = title_map.get(t.lower())
            merged.append({"title": t, "url": u})
        result["papers"] = merged
        return result
    except Exception:
        logging.exception("merge_with_arxiv_candidates failed")
        return result
    
async def timed_async(fn, *args, metric_key: str = None, **kwargs):
    """
    Run async function `fn` and record elapsed ms into app.state.metrics[metric_key].
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
        pass
    return res

async def remote_summary_extract(query: str, remote_url: Optional[str] = None, timeout_sec: int = 25) -> dict:
    """
    Calls the remote A2A summarizer and returns a validated dict:
    - On success: {"summary": "...", "papers":[{"title":"...","url":"..."},...]}
    - On failure: {"error": "...", "raw": "..."}
    """
    url = "http://localhost:9001/a2a"
    payload = {
        "invocation_id": str(uuid.uuid4()),
        "sender": "ResearchCoordinator",
        "query": query,
    }
    start = time.perf_counter()
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception as e:
        logging.exception("[remote_summary_extract] a2a call failed")
        try:
            app.state.metrics = getattr(app.state, "metrics", {})
            app.state.metrics["a2a_summarizer_ms"] = (time.perf_counter() - start) * 1000.0
            app.state.metrics["a2a_summarizer_fail_count"] = app.state.metrics.get("a2a_summarizer_fail_count", 0) + 1
        except Exception:
            pass
        return {"error": "remote_call_failed", "raw": str(e)}

    try:
        app.state.metrics = getattr(app.state, "metrics", {})
        app.state.metrics["a2a_summarizer_ms"] = (time.perf_counter() - start) * 1000.0
        app.state.metrics["a2a_summarizer_count"] = app.state.metrics.get("a2a_summarizer_count", 0) + 1
    except Exception:
        pass

    if not isinstance(data, dict):
        return {"error": "remote_nonjson_response", "raw": str(data)[:4000]}
    if not data.get("ok"):
        return {"error": "remote_ok_false", "raw": data}

    result = data.get("result")
    print("Raw result coming from payload", result)

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:

            result = {"error": "remote_result_not_json", "raw": result[:4000]}

    if not isinstance(result, dict):
        return {"error": "remote_result_bad_shape", "raw": str(result)[:4000]}

    if result.get("error"):
        return {"error": result.get("error"), "raw": result.get("raw") or result.get("raw", "")}
    if isinstance(result.get("summary"), str) and isinstance(result.get("papers"), list):
        papers = []
        for p in result.get("papers")[:5]:
            if not isinstance(p, dict):
                continue
            title = (p.get("title") or "").strip()
            url = (p.get("url") or p.get("link") or "").strip()
            if title and url:
                papers.append({"title": title, "url": url})
        return {"summary": result.get("summary").strip(), "papers": papers}
    return {"error": "remote_result_missing_fields", "raw": result}

async def summarizer(args: dict, tool_context=None):
    """
    ADK will call this with a dict payload. Normalize it and call your existing remote_summary_extract.
    Return both structured 'result' and 'content.parts[0].text' with a human-friendly summary + top-5 list.
    """
    logging.info("[summarizer] invoked; args_len=%d", len(str(args or "")))
    try:
        if not isinstance(args, dict):
            q = args if isinstance(args, str) else ""
            remote_url = None
            timeout_sec = 25
        else:
            q = args.get("query") or args.get("request") or args.get("prompt") or ""
            remote_url = args.get("remote_url") or None
            timeout_sec = int(args.get("timeout_sec", 25))

        result = await remote_summary_extract(q, remote_url=remote_url, timeout_sec=timeout_sec)

        if isinstance(result, dict) and result.get("summary"):
            sb = (result.get("summary") or "").strip()
            msg_lines = [f"Based on your query, here’s a quick summary:\n\n{sb}\n"]
            papers = result.get("papers") or []
            if papers:
                msg_lines.append("\nTop matched papers:")
                for i, p in enumerate(papers[:5], start=1):
                    title = (p.get("title") or "").strip() or "<untitled>"
                    url = (p.get("url") or p.get("pdf_url") or p.get("link") or "").strip()
                    if url:
                        msg_lines.append(f"{i}. {title} — {url}")
                    else:
                        msg_lines.append(f"{i}. {title}")
            msg = "\n".join(msg_lines)
        else:
            
            try:
                msg = json.dumps(result, ensure_ascii=False)
            except Exception:
                msg = str(result)

        return {
            "ok": True,
            "result": result,
            "content": {"parts": [{"text": msg}]}
        }

    except Exception as e:
        logging.exception("summarizer error")
        err = {"error": "wrapper_exception", "raw": str(e)}
        return {"ok": False, "result": err, "content": {"parts": [{"text": json.dumps(err, ensure_ascii=False)}]}}

# ---------------- Agent Declarations ---------------------

search_agent = LlmAgent(
    name="arxiv_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, api_key=GOOGLE_API_KEY),
    instruction="""
        Call the provided tool `arxiv_search` with the user's query and RETURN ONLY the tool/function RESPONSE OBJECT (no additional commentary or text).
        The tool response will be consumed by the next agent.
    """,
    tools=[arxiv_search, preload_tool],
    description="Search arXiv and write structured results to state",
    output_key="arxiv_candidates",
)


final_agent = LlmAgent(
    name="final_summarizer_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, api_key=GOOGLE_API_KEY),
    instruction="""
        You are final_summarizer_agent.
        Your job is to CALL the summarizer with a combined text (titles + abstracts) as 'query' to produce a structured JSON summary.
        Example call: {"tool":"summarizer","query":"Title1 - Abstract1\\n\\nTitle2 - Abstract2 ..."}
        WAIT for the remote summarizer's function_response (it will return structured JSON with keys 'summary', 'papers' and 'url').
        Extract the summary and links from summary agent and show the response (summary and papers including URLs) in the UI in a human readable format.
    """,
    tools=[
        FunctionTool(func=summarizer)
    ],
)

root_agent = SequentialAgent(
    name="ResearchCoordinator",
    sub_agents=[search_agent, final_agent],
    description="Sequentially run search_agent then final_summarizer_agent"
)

# ---------------- Lifespan ---------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create Pinecone client and index (defensive)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    try:
        if not pc.has_index(PINECONE_INDEX):
            try:
                pc.create_index_for_model(
                    name=PINECONE_INDEX,
                    cloud=PINECONE_CLOUD,
                    region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
                    embed={"model": PINECONE_MODEL, "field_map": {"text": "text"}},
                )
            except Exception:
                pc.create_index(
                    name=PINECONE_INDEX,
                    dimension=APP_EMBED_DIM,
                    metric="cosine"
                )
        # Describe index to derive any field_map
        info = pc.describe_index(PINECONE_INDEX) or {}
        field_map = {}
        try:
            if info.get("embed"):
                field_map = info["embed"].get("field_map") or {}
            if not field_map:
                field_map = info.get("spec", {}).get("integration", {}).get("embedding", {}).get("field_map", {}) or {}
        except Exception:
            field_map = {}
        if not field_map:
            field_map = {"text": "text"}

        # store runtime handles
        app.state.pinecone_client = pc
        app.state.dense_index = pc.Index(PINECONE_INDEX)
        app.state.field_map = field_map

        # minimal metrics slot
        app.state.metrics = getattr(app.state, "metrics", {})

        yield
    finally:
        pass

app.router.lifespan_context = lifespan

# ---------------- Routes ---------------------

@app.get("/")
def root():
    return {"ok": True, "service": "ResearcherAssistant", "note": "POST /search"}

@app.post("/search", response_model=SearchResponse)
async def search_papers(req: SearchRequest):
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="empty query")
    maxr = min(req.max_results or 10, 50)

    # shape for UI
    app.state.last_agent_output = getattr(app.state, "last_agent_output", {}) or {}
    app.state.last_agent_output.update({
        "raw_text": "",
        "summary": None,
        "papers": [],
        "links": [],
        "human_readable": None,
        "agent_final_text": "",
    })

    logging.info("HYBRID SEARCH q=%s maxr=%d", q, maxr)

    try:
        count_invocation_plugin = CountInvocationPlugin()
        app.state.count_invocation_plugin = count_invocation_plugin

        runner = Runner(
            agent=root_agent,
            app_name="agents",
            session_service=session_service,
            memory_service=memory_service,
            plugins=[LoggingPlugin(), app.state.count_invocation_plugin],
        )
        async def _run_and_persist_session(runner: Runner, session_obj, user_text: str):
            final_response_text = None
            events_seen = []
            try:
                content = types.Content(role="user", parts=[types.Part(text=user_text)])

                async for event in runner.run_async(
                    user_id=session_obj.user_id,
                    session_id=session_obj.id,
                    new_message=content,
                ):
                    # Collect a compact representation for debugging/inspection
                    try:
                        ev_repr = {
                            "id": getattr(event, "id", None),
                            "author": getattr(event, "author", None),
                            "type": getattr(event, "type", None),
                        }
                        # try to capture text content if present (defensive)
                        if getattr(event, "content", None) and getattr(event.content, "parts", None):
                            ev_repr["text_snippet"] = (event.content.parts[0].text or "")[:500]
                        events_seen.append(ev_repr)
                        logger.info("runner event: %s", ev_repr)
                    except Exception:
                        logger.exception("error serializing event")
                    
                    try:
                        # log last 10 full events for inspection (not just snippets)
                        logger.info("DEBUG: last runner events (full) for session=%s", session_obj.id)
                        # if runner yields event objects, write their reprs
                        for ev in events_seen[-20:]:
                            logger.info("DEBUG_EVENT: %s", ev)
                        # additionally, inspect app.state.last_agent_output
                        logger.info("DEBUG last_agent_output: %s", json.dumps(app.state.last_agent_output.get('agent_final_text') or "", ensure_ascii=False)[:4000])
                    except Exception:
                        logger.exception("failed to dump runner events")

                    # detect final response events (ADK event shapes vary; be defensive)
                    try:
                        if getattr(event, "is_final_response", None) and callable(event.is_final_response) and event.is_final_response():
                            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                                final_response_text = event.content.parts[0].text
                            break
                        # fallback heuristic: if event has .final == True or author == agent and content with text
                        if getattr(event, "final", False):
                            final_response_text = getattr(event.content.parts[0], "text", final_response_text) or final_response_text
                            break
                    except Exception:
                        logger.exception("error checking final response")

            except Exception:
                logger.exception("_run_and_persist_session: runner error")
            finally:
                # Persist session
                try:
                    if hasattr(memory_service, "add_session_to_memory"):
                        await memory_service.add_session_to_memory(session_obj)
                        logger.info("session persisted to memory: user_id=%s session_id=%s",
                                    session_obj.user_id, session_obj.id)
                    else:
                        logger.warning("memory_service missing add_session_to_memory; skipping persist")
                except Exception:
                    logger.exception("persisting session to memory failed (non-fatal)")

                # Attach events and final text to app.state.last_agent_output for UI/debug
                try:
                    cur = getattr(app.state, "last_agent_output", {}) or {}
                    cur.update({
                        "agent_events": events_seen[-20:],  # keep last 20 events
                        "agent_final_text": (final_response_text or cur.get("agent_final_text")),
                        "agent_session_id": getattr(session_obj, "id", None),
                    })
                    app.state.last_agent_output = cur
                    logger.info(
                        "attached runner events/final_text to last_agent_output (events=%d, final_present=%s)",
                        len(events_seen), bool(final_response_text)
                    )
                except Exception:
                    logger.exception("failed to attach runner events to last_agent_output")

                # safely inspect agent_final_text (run in finally block, after app.state.last_agent_output set)
                try:
                    final_text = app.state.last_agent_output.get("agent_final_text")
                    if final_text is None:
                        final_text = ""

                    # quick guard: skip parsing empty or obviously-non-json strings
                    ft_strip = final_text.strip()
                    if not ft_strip:
                        logger.info("FINAL_AGENT_JSON_PARSE_SKIPPED: agent_final_text empty")
                    elif not (ft_strip.startswith("{") or ft_strip.startswith("[")):
                        # not starting with JSON punctuation — still log snippet but skip heavy parse
                        snippet = (ft_strip[:200] + "...") if len(ft_strip) > 200 else ft_strip
                        logger.info("FINAL_AGENT_JSON_PARSE_SKIPPED: text does not start with JSON. snippet=%s", snippet)
                    else:
                        try:
                            parsed = json.loads(final_text)
                            logger.info("FINAL_AGENT_JSON_OK: keys=%s", list(parsed.keys()) if isinstance(parsed, dict) else "<non-dict-root>")
                        except Exception as decode_err:
                            snippet = (ft_strip[:1000] + "...") if ft_strip else "<empty>"
                            logger.warning("FINAL_AGENT_JSON_PARSE_FAILED: %s; snippet=%s", str(decode_err), snippet)
                except Exception:
                    logger.exception("Error while finalizing runner events (non-fatal)")

            return final_response_text

        # create a session explicitly so add_session_to_memory has a session object to operate on
        session = await session_service.create_session(app_name=runner.app_name or "agents", user_id=str(uuid.uuid4()))
        # schedule background run that will persist the session to memory when finished
        asyncio.create_task(_run_and_persist_session(runner, session, f"User query: {q}\n\nPlease follow the instructions carefully"))
    except Exception:
        logging.exception("Failed to start runner (non-fatal)")

    arxiv_candidates = await timed_async(arxiv_search, q, max_results=min(maxr, 10), metric_key="arxiv_search_ms")

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

    start = time.perf_counter()
    summarizer_result = await remote_summary_extract(combined, remote_url=None, timeout_sec=30)
    app.state.metrics = getattr(app.state, "metrics", {}) or {}
    app.state.metrics["summarizer_ms"] = (time.perf_counter() - start) * 1000.0

    try:
        summarizer_result = merge_with_arxiv_candidates(summarizer_result, arxiv_candidates)
    except Exception:
        logging.exception("merge_with_arxiv_candidates failed (non-fatal)")

    try:
        raw_text = json.dumps(summarizer_result, ensure_ascii=False)
    except Exception:
        raw_text = json.dumps({"error": "raw_text_serialization_failed", "repr": str(summarizer_result)[:2000]}, ensure_ascii=False)

    human_readable = None
    if summarizer_result.get("summary"):
        human_readable = "Here is the summary:\n\n" + summarizer_result["summary"] + "\n\n"
        if summarizer_result.get("papers"):
            human_readable += "Top papers:\n"
            for i, p in enumerate(summarizer_result["papers"][:5], start=1):
                human_readable += f"{i}. {p.get('title')} — {p.get('url')}\n"

    last_output = {
        "raw_text": raw_text,
        "summary": summarizer_result.get("summary"),
        "papers": summarizer_result.get("papers") or [],
        "links": summarizer_result.get("papers") or [],
        "human_readable": human_readable,
        "source": "hybrid-orchestrator",
    }
    app.state.last_agent_output = last_output
    logging.info("Published last_agent_output (summary_present=%s, papers=%d)",
                 bool(last_output.get("summary")), len(last_output.get("papers") or []))

    # upsert to Pinecone 
    try:
        upsert_recs = []
        for e in (arxiv_candidates or [])[:maxr]:
            upsert_recs.append({
                "id": f"arxiv::{e.get('id') or str(uuid.uuid4())}",
                "metadata": {"title": e.get("title"), "abstract": e.get("abstract"), "pdf_url": e.get("pdf_url")},
                "text": (e.get("title","") + "\n\n" + (e.get("abstract") or ""))[:3000],
            })
        if hasattr(app.state, "dense_index"):
            await asyncio.to_thread(prepare_and_upsert, app.state.dense_index, upsert_recs, "arxiv")
    except Exception:
        logging.exception("Pinecone upsert failed (non-fatal)")

    # build and return SearchResponse as before
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
        summary=last_output.get("summary"),
        top_papers=[p.dict() for p in papers_items[:5]],
        top_paper_links=[{"title": p.title, "url": p.pdf_url or p.arxiv_url or ""} for p in papers_items[:5]]
    )

@app.get("/admin/metrics")
def get_metrics():
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
    Return a small, UI-friendly blob:
      - summary (if available)
      - papers (if available) : expected list of {title, link, ...}
      - links (if available)
      - raw_text : agent raw string
    Frontend should render summary/papers if present, otherwise fall back to raw_text.
    """
    out = getattr(app.state, "last_agent_output", None) or {}
    return {
        "ok": True,
        "summary": out.get("summary"),
        "papers": out.get("papers"),
        "links": out.get("links"),
        "raw_text": out.get("raw_text"),
        "raw_blob": out,
    }

@app.get("/debug/watch_last_output")
def debug_watch_last_output(duration_sec: int = 10, poll_interval: float = 0.5):
    """
    Poll app.state for changes to last_agent_output and metrics for `duration_sec`.
    Returns timeline of observed values.
    """
    start = time.time()
    timeline = []
    seen = None
    while time.time() - start < duration_sec:
        try:
            s = getattr(app, "state", None)
            cur = getattr(s, "last_agent_output", None)
            metrics = getattr(s, "metrics", None)
            # string summary (short)
            summary = None
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


@app.post("/admin/clear_namespace")
async def clear_namespace(namespace: str = Body("arxiv", embed=True)):
    """
    DEV-ONLY: Delete all vectors in the given namespace (keeps the index).
    Pass JSON body: {"namespace": "arxiv"}.
    """
    try:
        idx = app.state.dense_index  # created in lifespan()
        # Attempt the typical SDK call that deletes all vectors in a namespace
        try:
           
            idx.delete(delete_all=True, namespace=namespace)
        except TypeError:
            
            idx.delete(delete_all=True)
        return {"ok": True, "note": f"cleared namespace {namespace}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)