import os
import uuid
import logging
from models.model import A2APayload
from fastapi import FastAPI, HTTPException
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin
from dotenv import load_dotenv
import asyncio
import json
import re

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("remote_summarizer")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="remote-summarizer", version="0.1")

# Local in-memory session service
session_service = InMemorySessionService()

# Define the remote summarizer agent
remote_summarizer_agent = LlmAgent(
    name="remote_summarizer_agent",
    model=Gemini(model=os.getenv("REMOTE_SUMMARIZER_MODEL", "gemini-2.5-flash-lite"), api_key=GOOGLE_API_KEY),
    instruction=(
        "You are a remote summarizer. GIVEN an input (titles+abstracts or docs), OUTPUT ONLY a single valid JSON object "
        "and nothing else. The JSON MUST have these keys:\n\n"
        "  {\"summary\": \"<1-3 sentence concise summary>\",\n"
        "   \"papers\": [ {\"title\":\"...\",\"url\":\"https://...\"}, ... up to 5 items ] }\n\n"
        "If you cannot create a proper summary or cannot find 5 papers, still RETURN JSON. Example valid output exactly:\n"
        "{\"summary\":\"Two-sentence summary.\", \"papers\":[{\"title\":\"T1\",\"url\":\"https://...\"}, {\"title\":\"T2\",\"url\":\"https://...\"}]}\n\n"
        "Do NOT include extra commentary, explanation, or formatting. If you fail, return: {\"error\":\"explain reason\"}."
    ),
)

def extract_text_from_events(events) -> str:
    """
    Extract plain text from a list of events returned by the Runner.
    """

    if events is None:
        return ""

    def try_parse_json_from_str(s: str):
        if not s or len(s) < 20:
            return None
        try:
            return json.loads(s)
        except Exception:
            pass
        cand = re.findall(r'(\{(?:.|\n)*\}|\[(?:.|\n)*\])', s)
        if not cand:
            return None
        cand.sort(key=len, reverse=True)
        for c in cand:
            try:
                return json.loads(c)
            except Exception:
                continue
        return None

    ev_list = events if isinstance(events, list) else [events]
    last_str_fallback = ""
    for ev in reversed(ev_list):
        try:
            c = getattr(ev, "content", None)
            if c:
                parts = getattr(c, "parts", None) or []

                for p in parts:
                    t = getattr(p, "text", None)
                    if t and isinstance(t, str) and t.strip():
                        return t.strip()

                for p in parts:
                    fr = getattr(p, "function_response", None)
                    if fr:
                        resp = getattr(fr, "response", None)
                        if isinstance(resp, dict):
                            for k in ("summary", "result_text", "result", "text", "body"):
                                if k in resp and isinstance(resp[k], str) and resp[k].strip():
                                    return resp[k].strip()
                            for v in resp.values():
                                if isinstance(v, str) and v.strip():
                                    return v.strip()
                        else:
                            s = str(resp)
                            if s and len(s) > 10:
                                parsed = try_parse_json_from_str(s)
                                if isinstance(parsed, dict) and parsed.get("summary"):
                                    return parsed.get("summary").strip()
                                return s.strip()
                for p in parts:
                    fc = getattr(p, "function_call", None)
                    if fc:
                        s = str(fc)
                        parsed = try_parse_json_from_str(s)
                        if isinstance(parsed, dict):
                            if parsed.get("summary"):
                                return parsed.get("summary").strip()
                            if parsed.get("papers") and parsed.get("summary"):
                                return parsed.get("summary").strip()
                        if s and len(s) > 10:
                            last_str_fallback = s
            if getattr(ev, "text", None):
                t = getattr(ev, "text")
                if t and isinstance(t, str) and t.strip():
                    return t.strip()
            s = str(ev)
            if s and len(s) > 10:
                last_str_fallback = s
        except Exception:
            continue

    if last_str_fallback:
        parsed = try_parse_json_from_str(last_str_fallback)
        if isinstance(parsed, dict) and parsed.get("summary"):
            return parsed.get("summary").strip()
        if "MALFORMED_FUNCTION_CALL" in last_str_fallback or "function_call" in last_str_fallback:
            try:
                sfull = str(events)
                full_parsed = try_parse_json_from_str(sfull)
                if isinstance(full_parsed, dict) and full_parsed.get("summary"):
                    return full_parsed.get("summary").strip()
            except Exception:
                pass
            return "remote_agent_malformed_function_call: summarizer attempted a tool-call but returned malformed function metadata."
        return last_str_fallback[:4000]

    return ""

@app.post("/a2a")
async def a2a_call(payload: A2APayload):
    if not payload or not payload.query:
        raise HTTPException(status_code=400, detail="query required")

    invocation_id = payload.invocation_id or str(uuid.uuid4())
    if payload.docs:
        parts = []
        for i, d in enumerate(payload.docs, 1):
            t = (d.get("title") or "")[:300]
            a = (d.get("abstract") or d.get("summary") or "")[:1200]
            parts.append(f"{i}. {t}\n{a}\n")
        prompt = (
            f"Summarize these {len(parts)} documents in 1-3 concise sentences and return JSON with keys 'summary' and 'papers' (list of title/url):\n\n"
            + "\n".join(parts)
        )
    else:
        # directly forward query to the agent instruction (the agent will be forced to JSON)
        prompt = payload.query

    runner = Runner(
        agent=remote_summarizer_agent,
        app_name="agents",
        session_service=session_service,
        plugins=[LoggingPlugin()],
    )

    res = runner.run_debug(prompt)
    if asyncio.iscoroutine(res):
        events = await res
    else:
        events = res

    result_text = extract_text_from_events(events) or ""
    def extract_json_from_text(s: str):
        if not s:
            return None
        try:
            j = json.loads(s)
            return j
        except Exception:
            cand = re.findall(r'(\{(?:.|\n)*\}|\[(?:.|\n)*\])', s)
            if not cand:
                return None
            cand.sort(key=len, reverse=True)
            for c in cand:
                try:
                    return json.loads(c)
                except Exception:
                    continue
            return None

    parsed = extract_json_from_text(result_text)
    if not parsed:
        parsed = extract_json_from_text(str(events))

    # Validate shape: must be dict and have either 'summary'+ 'papers' or an 'error'
    valid = False
    if isinstance(parsed, dict):
        if parsed.get("error") and isinstance(parsed.get("error"), str):
            valid = True
        elif isinstance(parsed.get("summary"), str) and isinstance(parsed.get("papers"), list):
            # enforce up to 5 papers and each has title+url
            papers = parsed.get("papers")[:5]
            cleaned = []
            for p in papers:
                if not isinstance(p, dict):
                    continue
                title = (p.get("title") or "").strip()
                url = (p.get("url") or p.get("link") or "").strip()
                if title and url:
                    cleaned.append({"title": title, "url": url})
            parsed["papers"] = cleaned
            valid = True

    if not valid:
        # return an error shape to calling service so they always receive JSON
        parsed = {"error": "remote_summarizer_malformed_output", "raw": result_text[:4000]}

    logger.info("[a2a] invocation_id=%s sender=%s result_keys=%s", invocation_id, payload.sender, list(parsed.keys()))
    return {"ok": True, "invocation_id": invocation_id, "result": parsed}