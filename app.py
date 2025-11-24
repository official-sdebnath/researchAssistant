from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.function_tool import FunctionTool

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

# ------------------------ Config ------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "researcher-assistant")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_MODEL = os.getenv("PINECONE_MODEL", "llama-text-embed-v2")
APP_EMBED_DIM = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment. Add to .env")

ARXIV_API = "http://export.arxiv.org/api/query"

app = FastAPI(
    title="Researcher Assistant",
    version="minimal-fixed-upsert"
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
    Counts agent runs and LLM requests, records timings, and writes simple metrics to app.state.metrics.
    """
    def __init__(self) -> None:
        super().__init__(name="count_invocation")
        self.agent_count = 0
        self.llm_request_count = 0
        self.last_agent_run_ms = None
        self.last_llm_request_ms = None

    async def before_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> None:
        callback_context._agent_start = time.perf_counter()
        self.agent_count += 1
        logging.info(f"[CountInvocationPlugin] Agent run #{self.agent_count} for agent={getattr(agent, 'name', None)}")

    async def after_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext, result: Any = None) -> None:
        try:
            elapsed = max(0.0, (time.perf_counter() - getattr(callback_context, "_agent_start", time.perf_counter())) * 1000.0)
            self.last_agent_run_ms = elapsed
            logging.info(f"[CountInvocationPlugin] Agent {getattr(agent, 'name', None)} finished in {elapsed:.1f} ms")
            try:
                app.state.metrics = getattr(app.state, "metrics", {})
                app.state.metrics["last_agent_run_ms"] = elapsed
                app.state.metrics["agent_count"] = self.agent_count
            except Exception:
                pass
        except Exception:
            logging.exception("[CountInvocationPlugin] after_agent_callback error")

    async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        callback_context._llm_start = time.perf_counter()
        self.llm_request_count += 1
        logging.info(f"[CountInvocationPlugin] LLM request #{self.llm_request_count}; prompt_len={len(getattr(llm_request, 'prompt', '') or '')}")

    async def after_model_callback(self, *, callback_context: CallbackContext, llm_response: Any = None) -> None:
        try:
            elapsed = max(0.0, (time.perf_counter() - getattr(callback_context, "_llm_start", time.perf_counter())) * 1000.0)
            self.last_llm_request_ms = elapsed
            logging.info(f"[CountInvocationPlugin] LLM response received in {elapsed:.1f} ms")
            try:
                app.state.metrics = getattr(app.state, "metrics", {})
                app.state.metrics["last_llm_request_ms"] = elapsed
                app.state.metrics["llm_request_count"] = self.llm_request_count
            except Exception:
                pass
        except Exception:
            logging.exception("[CountInvocationPlugin] after_model_callback error")

# ---------------- Pydantic Models ---------------------

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: Optional[int] = Field(10, ge=1, le=50)

class PaperItem(BaseModel):
    id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    pdf_url: Optional[str] = None
    published: Optional[datetime] = None
    arxiv_url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    count: int
    papers: List[PaperItem]
    summary: Optional[str] = None
    top_papers: Optional[List[PaperItem]] = None
    top_paper_links: Optional[List[Dict[str, str]]] = None 

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
    Upserts using the index's field_map to pick the text field name.
    Builds records like: {"_id": "...", "<text_field>": "...", <meta fields> }
    and calls dense_index.upsert_records(namespace, records_payload).
    Also mirrors upserted records into app.state._local_index[namespace] for cheap local retrieval.
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
        # Merge metadata as top-level keys
        for k, v in meta.items():
            if k in ("_id", text_field):
                r[f"meta_{k}"] = v
            else:
                r[k] = v
        records_payload.append(r)

    # ------------------ Mirror into local index ------------------
    try:
        local = getattr(app.state, "_local_index", None)
        if local is None:
            app.state._local_index = {}
            local = app.state._local_index
        ns_list = local.get(namespace, [])
        existing_ids = {r.get("_id") for r in ns_list}
        for r in records_payload:
            if r.get("_id") not in existing_ids:
                ns_list.append(r)
        local[namespace] = ns_list
    except Exception:
        pass

    try:
        # signature: upsert_records (namespace, records)
        return dense_index.upsert_records(namespace, records_payload) or {"ok": True, "count": len(records_payload)}
    except Exception as e:
        print("Pinecone upsert failed:", str(e))
        return {"ok": False, "error": str(e)}

def retrieve_from_pinecone(query: str, top_k: int = 5, namespace: str = "arxiv") -> List[dict]:
    """
    Picks the best text field available (prefers 'text', then field_map's text field, then title, then first string).
    Returns list of dicts: {id, text, metadata}
    """
    if not query:
        return []

    local = getattr(app.state, "_local_index", {}) or {}
    candidates = local.get(namespace, []) 
    if not candidates:
        return []

    def pick_text(rec: Dict[str, Any]) -> str:

        if isinstance(rec.get("text"), str) and rec.get("text"):
            return rec["text"]
    
        field_map = getattr(app.state, "field_map", {}) or {}
        text_field = field_map.get("text")
        if text_field and isinstance(rec.get(text_field), str) and rec.get(text_field):
            return rec[text_field]
       
        if isinstance(rec.get("title"), str) and rec.get("title"):
            return rec["title"]

        for k, v in rec.items():
            if k == "_id" or k.startswith("meta_"):
                continue
            if isinstance(v, str) and v:
                return v
        
        pieces = [v for k, v in rec.items() if isinstance(v, str) and k != "_id"]
        return " ".join(pieces)[:3000]

    q = query.lower().strip()
    scored = []
    for rec in candidates:
        text = (pick_text(rec) or "").lower()
        title = (rec.get("title") or "").lower()
        score = 0
        if q in text:
            score += 3
        if q in title:
            score += 4
        q_words = set(q.split())
        overlap = len(q_words.intersection(set((text + " " + title).split())))
        score += overlap * 0.2
        scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [r for s, r in scored][:top_k]
    out = []
    for r in results:
        out.append({
            "id": r.get("_id"),
            "text": pick_text(r),
            "metadata": {k: v for k, v in r.items() if k not in ("_id",)}
        })
    return out

# ------------------ Generic Helpers ------------------
    
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

async def remote_summarizer_tool(query: str, remote_url: Optional[str] = None, timeout_sec: int = 25) -> dict:
    """
    Calls the remote A2A summarizer and returns a validated dict:
    - On success: {"summary": "...", "papers":[{"title":"...","url":"..."},...]}
    - On failure: {"error": "...", "raw": "..."}
    """
    url = remote_url or os.getenv("http://localhost:9001/a2a")
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
        logging.exception("[remote_summarizer_tool] a2a call failed")
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


async def remote_summarizer_adk_tool(args: dict, tool_context=None):
    """
    ADK will call this with a dict payload. Normalize it and call your existing remote_summarizer_tool.
    Return both structured 'result' and 'content.parts[0].text' with a human-friendly summary + top-5 list.
    """
    logging.info("[remote_summarizer_adk_tool] invoked; args_len=%d", len(str(args or "")))
    try:
        if not isinstance(args, dict):
            q = args if isinstance(args, str) else ""
            remote_url = None
            timeout_sec = 25
        else:
            q = args.get("query") or args.get("request") or args.get("prompt") or ""
            remote_url = args.get("remote_url") or None
            timeout_sec = int(args.get("timeout_sec", 25))

        result = await remote_summarizer_tool(q, remote_url=remote_url, timeout_sec=timeout_sec)

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
        logging.exception("remote_summarizer_adk_tool error")
        err = {"error": "wrapper_exception", "raw": str(e)}
        return {"ok": False, "result": err, "content": {"parts": [{"text": json.dumps(err, ensure_ascii=False)}]}}

# ---------------- Agent Declarations ---------------------

search_agent = LlmAgent(
    name="arxiv_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, api_key=GOOGLE_API_KEY),
    instruction="""
    You are an assistant whose job is to return a machine-readable list of arXiv papers for the given query uisng the assigned tool. 
    After gathering the relevant papers, respond with
    **SEARCH_COMPLETE**
""",
    tools=[arxiv_search],
)

root_agent = LlmAgent(
    name="ResearchCoordinator",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config, api_key=GOOGLE_API_KEY),
    instruction="""
        You are ResearchCoordinator.
        Workflow (STRICT):
        1) CALL the tool search_agent with the user's query to gather candidate papers.
        2) WAIT for its function_response containing titles & abstracts.
        3a) After search_agent gives results (SEARCH_COMPLETE) or FINISHED, ensure you move to the second phase: CALL the remote_summarizer_tool. NEVER miss or forget to call remote_summarizer_tool.
        3b) CALL the tool remote_summarizer_tool with a combined text (titles + abstracts) as 'query' to produce a structured JSON summary.
           Example call: {"tool":"remote_summarizer_tool","query":"Title1 - Abstract1\\n\\nTitle2 - Abstract2 ..."}
        4) WAIT for the remote summarizer's function_response (it will return structured JSON with keys 'summary' and 'papers').
        5) Extract the summary and links from summary agent and show the response (summary and papers) in the UI in a human readable format.
 
            """,
    tools=[
        AgentTool(search_agent),
        FunctionTool(func=remote_summarizer_adk_tool)
    ],
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
        app.state._local_index = getattr(app.state, "_local_index", {})

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
    })

    logging.info("HYBRID SEARCH q=%s maxr=%d", q, maxr)

    try:
        runner = Runner(
            agent=root_agent,
            app_name="agents",
            session_service=session_service,
            plugins=[LoggingPlugin(), CountInvocationPlugin()],
        )
        asyncio.create_task(runner.run_debug(f"User query: {q}\n\n{"Please follow the instrcutions carefully"}"))
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

    summarizer_result = await remote_summarizer_tool(combined, remote_url=None, timeout_sec=30)

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
    """
    Simple in-app metrics for reviewers: returns counts and last durations.
    """
    try:
        metrics = getattr(app.state, "metrics", {}) or {}
        resp = {
            "agent_count": metrics.get("agent_count"),
            "llm_request_count": metrics.get("llm_request_count"),
            "last_agent_run_ms": metrics.get("last_agent_run_ms"),
            "last_llm_request_ms": metrics.get("last_llm_request_ms"),
            "arxiv_search_ms": metrics.get("arxiv_search_ms"),
            "upsert_ms": metrics.get("upsert_ms"),
            "summarizer_ms": metrics.get("summarizer_ms")
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
    
        if hasattr(app.state, "_local_index"):
            app.state._local_index.pop(namespace, None)
        return {"ok": True, "namespace": namespace, "note": "delete_all issued; local mirror cleared"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)