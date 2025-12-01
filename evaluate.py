import json
import time
import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List

# Evaluator for the Researcher Assistant project.
# Runs the root_agent on a small eval set and writes a JSON report.

import app

ROOT = Path(__file__).parent

# Load eval set
evalset_path = ROOT / "integration.evalset.json"
with open(evalset_path, "r") as f:
    evalset = json.load(f)

# Criteria for evaluation
config_path = ROOT / "test_config.json"
if config_path.exists():
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    config = {}

# Symbols pulled from app.py
root_agent = app.root_agent
Runner = app.Runner
session_service = app.session_service


# ---------------- Utility helpers ----------------

def safe_parse_json(text: str) -> Any:
    """Try json.loads; if that fails, pull the largest JSON-looking block and parse that."""
    if not text:
        return None

    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    # Extract any {...} or [...] spans and try the largest first
    candidates = re.findall(r'(\{(?:.|\n)*\}|\[(?:.|\n)*\])', s)
    if not candidates:
        return None

    candidates.sort(key=len, reverse=True)
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def similarity_score(a: str, b: str) -> float:
    """Token-overlap similarity on lowercased words."""
    a_s, b_s = (a or "").strip().lower(), (b or "").strip().lower()
    if not a_s and not b_s:
        return 1.0
    if not a_s or not b_s:
        return 0.0

    a_tokens = set(a_s.split())
    b_tokens = set(b_s.split())
    if not b_tokens:
        return 0.0

    return len(a_tokens.intersection(b_tokens)) / len(b_tokens)


def extract_text_from_events(events) -> str:
    """
    Pull the most useful text out of a Google ADK event sequence.

    Priority (from last event backwards):
      1) content.parts[*].text
      2) event.text
      3) str(last event)
    """
    if not events:
        return ""

    ev_list = events if isinstance(events, list) else [events]

    for ev in reversed(ev_list):
        try:
            content = getattr(ev, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None) or []
                for p in parts:
                    txt = getattr(p, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()

            ev_text = getattr(ev, "text", None)
            if isinstance(ev_text, str) and ev_text.strip():
                return ev_text.strip()
        except Exception:
            continue

    # Last resort: string form of the last event
    try:
        return str(ev_list[-1])
    except Exception:
        return ""


# ---------------- Core evaluation logic ----------------

async def run_case(case: Dict) -> Dict:
    """
    Run one eval case:
      - Send the user message through root_agent via Runner.run_debug(...)
      - Extract final text from events
      - Compare against the expected text
    """
    conv = case["conversation"][0]

    user_msg = conv["user_content"]["parts"][0]["text"]
    expected_text = ""
    if "final_response" in conv:
        parts = conv["final_response"].get("parts", [])
        if parts:
            expected_text = parts[0].get("text", "") or ""

    runner = Runner(agent=root_agent, app_name="agents", session_service=session_service)

    start = time.perf_counter()
    try:
        maybe = runner.run_debug(user_msg)
        # run_debug can be sync or async depending on ADK version
        events = await maybe if asyncio.iscoroutine(maybe) else maybe
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "eval_id": case.get("eval_id"),
            "user_msg": user_msg,
            "error": str(e),
            "elapsed_sec": elapsed,
            "passed": False,
        }
    elapsed = time.perf_counter() - start

    actual_text = extract_text_from_events(events) or ""

    # If the agent returns JSON with a "summary" field, compare on that
    parsed = safe_parse_json(actual_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("summary"), str) and parsed["summary"].strip():
        compare_text = parsed["summary"].strip()
    else:
        compare_text = actual_text

    resp_threshold = float(config.get("criteria", {}).get("response_match_score", 0.5))
    resp_score = similarity_score(compare_text, expected_text)
    passed = resp_score >= resp_threshold

    return {
        "eval_id": case.get("eval_id"),
        "user_msg": user_msg,
        "expected_text": expected_text,
        "actual_text": actual_text,
        "compare_text": compare_text,
        "response_similarity": resp_score,
        "threshold": resp_threshold,
        "elapsed_sec": elapsed,
        "passed": bool(passed),
    }


async def main():
    results: List[Dict] = []

    for case in evalset.get("eval_cases", []):
        print("Running:", case.get("eval_id"))
        try:
            r = await run_case(case)
            results.append(r)

            status = "PASS" if r["passed"] else "FAIL"
            sim = r.get("response_similarity", 0.0)
            thr = r.get("threshold", 0.0)
            print(
                f"  [{status}] resp_sim={sim:.3f} (threshold={thr:.3f}), "
                f"time={r['elapsed_sec']:.2f}s"
            )
        except Exception as e:
            print("  [ERROR]", e)
            results.append({"eval_id": case.get("eval_id"), "error": str(e)})

    # Simple pass/fail summary
    passed_cases = [r for r in results if r.get("passed")]
    failed_cases = [r for r in results if r.get("passed") is False]

    print("\n=== Evaluation Summary ===")
    print(f"Total cases: {len(results)}")
    print(f"Passed     : {len(passed_cases)}")
    print(f"Failed     : {len(failed_cases)}")

    if passed_cases:
        print("  Passed eval_ids:", ", ".join(r.get("eval_id", "?") for r in passed_cases))
    if failed_cases:
        print("  Failed eval_ids:", ", ".join(r.get("eval_id", "?") for r in failed_cases))

    out = {
        "run_at": time.ctime(),
        "results": results,
        "config": config,
    }

    out_path = ROOT / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved report to", out_path)


if __name__ == "__main__":
    asyncio.run(main())