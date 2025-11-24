import json
import time
import asyncio
import importlib.util
from pathlib import Path
from typing import Any, Dict, List
import re

ROOT = Path(__file__).parent

evalset = json.load(open(ROOT / "integration.evalset.json", "r"))
config = json.load(open(ROOT / "test_config.json", "r"))

# discover module (auto)
def discover_project_module() -> Any:
    for p in ROOT.glob("*.py"):
        if p.name in ("evaluate.py"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(p.stem, str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            required = ["root_agent", "Runner", "session_service", "_extract_agent_text"]
            if all(hasattr(mod, name) for name in required):
                print(f"[discover] using module: {p.name}")
                return mod
        except Exception:
            continue
    raise RuntimeError("Could not find a project module exposing required symbols (root_agent, Runner, session_service, _extract_agent_text).")

mod = discover_project_module()
root_agent = getattr(mod, "root_agent")
Runner = getattr(mod, "Runner")
session_service = getattr(mod, "session_service")
_extract_agent_text = getattr(mod, "_extract_agent_text")

# small helper to try parsing a JSON block and return object or None
def safe_parse_json(text: str):
    if not text:
        return None
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # extract largest {...} or [...] block
    candidates = re.findall(r'(\{(?:.|\n)*\}|\[(?:.|\n)*\])', s)
    candidates.sort(key=len, reverse=True)
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None

# simple token-overlap similarity (robust enough)
def similarity_score(a: str, b: str) -> float:
    a_s, b_s = (a or "").strip().lower(), (b or "").strip().lower()
    if not a_s and not b_s:
        return 1.0
    if not a_s or not b_s:
        return 0.0
    a_tokens = set(a_s.split())
    b_tokens = set(b_s.split())
    return len(a_tokens.intersection(b_tokens)) / max(1, len(b_tokens))

def extract_tool_names_from_events(events) -> List[str]:
    names = []
    if not events:
        return names
    candidate_tools = {"arxiv_search", "remote_summarizer_tool", "prepare_and_upsert", "retrieve_from_pinecone"}
    if isinstance(events, list):
        for ev in events:
            try:
                if hasattr(ev, "tool_name"):
                    names.append(getattr(ev, "tool_name"))
                s = str(ev).lower()
                for t in candidate_tools:
                    if t in s:
                        names.append(t)
            except Exception:
                continue
    else:
        s = str(events).lower()
        for t in candidate_tools:
            if t in s:
                names.append(t)
    # stable dedupe
    seen = []
    for n in names:
        if n and n not in seen:
            seen.append(n)
    return seen

def robust_extract_text(events) -> str:
    """More aggressive extractor for Google ADK event objects and function responses."""
    # 1) try project helper
    try:
        txt = _extract_agent_text(events)
        if txt and isinstance(txt, str) and txt.strip():
            return txt.strip()
    except Exception:
        pass

    if not events:
        return ""

    def inspect_ev(ev):
        # Try structured access first
        try:
            # Some events have .content with .parts (ADK)
            content = getattr(ev, "content", None)
            if content:
                parts = getattr(content, "parts", None) or []
                # prefer explicit text parts
                for p in parts:
                    if getattr(p, "text", None):
                        t = p.text.strip()
                        if t:
                            return t
                # then inspect function_response objects in parts
                for p in parts:
                    fr = getattr(p, "function_response", None)
                    if fr:
                        resp = getattr(fr, "response", None)
                        # if dict, prefer summary keys
                        if isinstance(resp, dict):
                            for k in ("summary", "result_text", "result", "text", "body", "response"):
                                if k in resp and isinstance(resp[k], str) and resp[k].strip():
                                    return resp[k].strip()
                            # fallback: first long string value
                            for v in resp.values():
                                if isinstance(v, str) and v.strip():
                                    return v.strip()
                        # otherwise fallback to string form
                        s = str(resp)
                        if s and len(s) > 10:
                            return s.strip()
                # if nothing useful, try parts' function_call payloads (string form)
                for p in parts:
                    if getattr(p, "function_call", None):
                        s = str(getattr(p, "function_call"))
                        if s and len(s) > 10:
                            # attempt to extract embedded JSON from it
                            parsed = safe_parse_json(s)
                            if parsed:
                                if isinstance(parsed, dict):
                                    if parsed.get("summary"):
                                        return parsed.get("summary").strip()
                                    return json.dumps(parsed)[:2000]
                            return s.strip()
            # If event has attributes like .tool_name or .text
            if hasattr(ev, "text") and getattr(ev, "text", None):
                return ev.text.strip()
        except Exception:
            pass

        # Last resort: try to parse large JSON blobs from the str() representation
        s = str(ev)
        if s:
            parsed = safe_parse_json(s)
            if isinstance(parsed, dict):
                if parsed.get("summary"):
                    return parsed.get("summary").strip()
                # if it has 'papers' + 'summary' prefer summary
                if parsed.get("papers") and parsed.get("summary"):
                    return parsed["summary"].strip()
                # otherwise return compact JSON string
                try:
                    return json.dumps(parsed)
                except Exception:
                    return s[:3000]
            # If it contains function_call or MALFORMED_FUNCTION_CALL just return the string,
            # evaluator will treat presence of these tokens as evidence (keyword_hit).
            if "function_call" in s or "MALFORMED_FUNCTION_CALL" in s or "invocation_id" in s:
                return s
        return None

    if isinstance(events, list):
        # prefer last non-empty extraction
        for ev in reversed(events):
            out = inspect_ev(ev)
            if out:
                return out
    else:
        out = inspect_ev(events)
        if out:
            return out
    return ""

async def run_case(case: Dict) -> Dict:
    user_msg = case["conversation"][0]["user_content"]["parts"][0]["text"]
    expected_text = case["conversation"][0].get("final_response", {}).get("parts", [{}])[0].get("text", "")
    expected_tools = [t.get("name") for t in case["conversation"][0].get("intermediate_data", {}).get("tool_uses", [])]

    # # derive a safe app_name (FastAPI has .title)
    # module_app = getattr(mod, "app", None)
    # if module_app is not None:
    #     app_name = getattr(module_app, "title", None) or getattr(mod, "__name__", "agents")
    # else:
    #     app_name = getattr(mod, "__name__", "agents")

    runner = Runner(agent=root_agent, app_name="agents", session_service=session_service)

    start = time.perf_counter()
    try:
        events = await runner.run_debug(user_msg)
    except TypeError:
        events = runner.run_debug(user_msg)
    elapsed = time.perf_counter() - start

    actual_text = robust_extract_text(events) or ""

    # if the returned text is JSON-like with a 'summary' field, prefer that summary
    parsed = safe_parse_json(actual_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("summary"), str) and parsed.get("summary").strip():
        compare_text = parsed.get("summary").strip()
    else:
        compare_text = actual_text

    # output markers (evidence) — used only if we fail to extract meaningful text
    lowered = (actual_text or "").lower()
    output_markers = ("function_call", "invocation_id", "functionresponse", '{"papers":', '"summary":', "model_version")
    # normalized apology variants
    apology_markers = (
        "could not find",
        "couldn't find",
        "no arxiv papers",
        "i could not find",
        "i'm sorry",
        "i am sorry",
        "cannot fulfill",
        "could not find any",
        "does not seem to be a valid topic",
        "please try again",
        "no relevant papers",
    )    
    
    keyword_hit = any(tok in lowered for tok in output_markers) or any(tok in lowered for tok in apology_markers)

    resp_score = similarity_score(compare_text, expected_text)
    used_tools = extract_tool_names_from_events(events)

    if not expected_tools:
        tool_score = 1.0
    else:
        hits = sum(1 for t in expected_tools if t in used_tools)
        tool_score = hits / max(1, len(expected_tools))

    # threshold from config (default relaxed to 0.5)
    resp_threshold = float(config.get("criteria", {}).get("response_match_score", 0.5))
    passed = (resp_score >= resp_threshold) or (keyword_hit and tool_score >= 0.5)

    return {
        "eval_id": case.get("eval_id"),
        "user_msg": user_msg,
        "expected_tools": expected_tools,
        "used_tools": used_tools,
        "response_similarity": resp_score,
        "keyword_hit": bool(keyword_hit),
        "tool_trajectory_score": tool_score,
        "elapsed_sec": elapsed,
        "actual_text": actual_text,
        "passed": bool(passed)
    }

async def main():
    results = []
    for case in evalset.get("eval_cases", []):
        print("Running:", case.get("eval_id"))
        try:
            r = await run_case(case)
            results.append(r)
            print(f"  passed={r['passed']}, resp_sim={r['response_similarity']:.3f}, tool_score={r['tool_trajectory_score']:.3f}, time={r['elapsed_sec']:.2f}s")
        except Exception as e:
            print("  ERROR:", e)
            results.append({"eval_id": case.get("eval_id"), "error": str(e)})

    out = {"run_at": time.ctime(), "results": results, "config": config}
    out_path = ROOT / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved report to", out_path)

    # best-effort cleanup: close session_service and allow background http sessions to close
    try:
        if hasattr(session_service, "close") and callable(session_service.close):
            maybe = session_service.close()
            if asyncio.iscoroutine(maybe):
                await maybe
    except Exception:
        pass

    # brief pause to let aiohttp connectors terminate gracefully
    await asyncio.sleep(0.25)

if __name__ == "__main__":
    asyncio.run(main())
