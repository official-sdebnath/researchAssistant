import os
import uuid
import json
import asyncio
import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from models.model import A2APayload
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.models.google_llm import Gemini

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("remote_summarizer")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="remote-summarizer", version="0.1")

# Session service used by the remote LLM agent
session_service = InMemorySessionService()


# ---------------------------------------------------------------------------
# Remote Summarizer Agent
# ---------------------------------------------------------------------------
# This agent receives document metadata (titles + abstracts) and produces
# a short 1–3 sentence summary. It must return either:
#   - A JSON object with a "summary" field, OR
#   - Plain text containing only the summary.
# No conversational filler or metadata is allowed.
# ---------------------------------------------------------------------------

remote_summarizer_agent = LlmAgent(
    name="remote_summarizer_agent",
    model=Gemini(
        model=os.getenv("REMOTE_SUMMARIZER_MODEL", "gemini-2.5-flash-lite"),
        api_key=GOOGLE_API_KEY
    ),
    instruction=(
        "Summarize the given documents or text. Output ONLY either:\n"
        "  1) A JSON object with a 'summary' field (and optionally 'papers'), OR\n"
        "  2) Plain text with a concise 1–3 sentence summary.\n"
        "Do not include explanations or conversation—only the summary."
    ),
)


# ---------------------------------------------------------------------------
# Utility: Try to extract JSON from a raw text LLM output
# ---------------------------------------------------------------------------
def extract_json_from_text(s: str) -> Optional[Any]:
    if not s:
        return None

    # Try direct JSON parsing first
    try:
        return json.loads(s)
    except Exception:
        pass

    # Fallback: extract the largest JSON-looking substring from the text
    import re
    candidates = re.findall(r'(\{(?:.|\n)*\}|\[(?:.|\n)*\])', s)
    if not candidates:
        return None

    candidates.sort(key=len, reverse=True)

    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue

    return None


# ---------------------------------------------------------------------------
# Utility: Extract human-readable text from ADK event objects
# ---------------------------------------------------------------------------
def extract_text_from_events(events) -> str:
    """
    Returns the most useful text fragment from the ADK event stream.
    Priority:
        1. content.parts[*].text
        2. content.parts[*].function_response.response (string or dict)
        3. event.text
        4. string representation of the event
    """
    if events is None:
        return ""

    event_list = events if isinstance(events, list) else [events]

    for ev in reversed(event_list):
        try:
            content = getattr(ev, "content", None)
            if content:
                parts = getattr(content, "parts", []) or []

                # Extract plain text from parts
                for p in parts:
                    if isinstance(getattr(p, "text", None), str) and p.text.strip():
                        return p.text.strip()

                # Extract function response payloads
                for p in parts:
                    fr = getattr(p, "function_response", None)
                    if fr:
                        resp = getattr(fr, "response", None)
                        if isinstance(resp, str) and resp.strip():
                            return resp.strip()
                        if isinstance(resp, dict):
                            if isinstance(resp.get("summary"), str):
                                return resp.get("summary").strip()
                            return json.dumps(resp, ensure_ascii=False)

            # If event has a direct text field
            if isinstance(getattr(ev, "text", None), str) and ev.text.strip():
                return ev.text.strip()

            # Fallback: string representation
            s = str(ev)
            if s and len(s) > 10:
                return s.strip()

        except Exception:
            continue

    return ""


# ---------------------------------------------------------------------------
# A2A Endpoint
# ---------------------------------------------------------------------------
# Receives: { query: str, docs: [...] }
# Returns:  { ok: True, invocation_id, result: { summary, papers } }
# ---------------------------------------------------------------------------

@app.post("/a2a")
async def a2a_call(payload: A2APayload):
    if not payload or not (payload.query or payload.docs):
        raise HTTPException(status_code=400, detail="query or docs required")

    invocation_id = payload.invocation_id or str(uuid.uuid4())

    logger.info(
        "[a2a] invocation_id=%s sender=%s query_len=%d docs_len=%d",
        invocation_id,
        payload.sender,
        len(payload.query or ""),
        len(payload.docs or []),
    )

    # -------------------------------------------------------------------
    # Build summarization prompt:
    # If docs exist → use titles + abstracts
    # If not → fall back to raw query text.
    # -------------------------------------------------------------------
    if payload.docs:
        parts = []
        for i, d in enumerate(payload.docs, start=1):
            title = (d.get("title") or "")[:300]
            abstract = (d.get("abstract") or d.get("summary") or "")[:1200]
            parts.append(f"{i}. {title}\n{abstract}\n")

        prompt = (
            f"Summarize these {len(parts)} documents in 1–3 sentences. "
            f"Return ONLY a JSON summary or plain text summary:\n\n"
            + "\n".join(parts)
        )
    else:
        prompt = payload.query

    # Runner for the remote summarizer agent
    runner = Runner(
        agent=remote_summarizer_agent,
        app_name="agents",
        session_service=session_service,
        plugins=[LoggingPlugin()],
    )

    # -------------------------------------------------------------------
    # Execute summarizer agent and extract usable text
    # -------------------------------------------------------------------
    events = None
    result_text = ""

    try:
        result = runner.run_debug(prompt)
        events = await result if asyncio.iscoroutine(result) else result
        result_text = extract_text_from_events(events) or ""
        logger.info("[a2a] extracted result_text length=%d", len(result_text))

    except Exception as e:
        # Fail gracefully — return an empty summary but preserve docs
        logger.exception(
            "[a2a] runner failed — returning empty summary (invocation_id=%s). Error: %s",
            invocation_id, str(e)
        )
        result_text = ""

    # -------------------------------------------------------------------
    # Parse result: JSON → summary, or fallback to plain text
    # -------------------------------------------------------------------
    parsed = extract_json_from_text(result_text)
    summary = ""

    if isinstance(parsed, dict) and isinstance(parsed.get("summary"), str):
        summary = parsed.get("summary").strip()
    elif result_text.strip():
        summary = result_text.strip()

    # Always return the original document list
    papers = payload.docs or []

    logger.info(
        "[a2a] invocation_id=%s returning summary_len=%d papers_count=%d",
        invocation_id,
        len(summary or ""),
        len(papers)
    )

    return {
        "ok": True,
        "invocation_id": invocation_id,
        "result": {"summary": summary, "papers": papers},
    }
