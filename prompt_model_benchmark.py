"""Run Phase 3 prompt tests and Phase 4 provider/model comparisons.

Set the needed environment variables, then run:
    python prompt_model_benchmark.py

The defaults run all four prompt strategies against every configured model.
Use --models groq_qwen --limit 5 for an inexpensive Phase 3 smoke test.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ADDED: Imported JUDGE_PROMPT and judge_count for Task 10
from evaluation import Prediction, evaluate, extract_json, load_dataset, string_list, JUDGE_PROMPT, judge_count

DATASET = Path(__file__).with_name("test_dataset.json")
LOG_DIR = Path(__file__).with_name("logs")

def load_local_env() -> None:
    """Load KEY=VALUE pairs from benchmark.env, overwriting any existing environment variables."""
    path = Path(__file__).with_name("benchmark.env")
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_local_env()

EXAMPLE_ONE = """Transcript: Maya: Leo will send the security review by Friday. Maya: An unsigned certificate could block launch.
Output: {"tasks":["Send the security review by Friday (Assignee: Leo)"],"risks":["Launch blocked by an unsigned certificate"]}"""
EXAMPLE_TWO = """Transcript: Omar: Priya will update the runbook today. Omar: The backup has not been restored successfully.
Output: {"tasks":["Update the runbook today (Assignee: Priya)"],"risks":["Backup recovery failure"]}"""

@dataclass(frozen=True)
class ModelProfile:
    key: str
    provider: str
    model: str
    api_key_env: str
    input_per_million_usd: float
    output_per_million_usd: float
    native_rate_delay: bool = False

PROFILES = {
    "gpt_4_1": ModelProfile("gpt_4_1", "openai", os.getenv("OPENAI_MODEL", "gpt-4.1"), "OPENAI_API_KEY", 2.00, 8.00),
    "gemini_2_5_flash": ModelProfile("gemini_2_5_flash", "gemini", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), "GEMINI_API_KEY", 0.30, 2.50, True),
    "claude_sonnet": ModelProfile("claude_sonnet", "anthropic", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"), "ANTHROPIC_API_KEY", 3.00, 15.00, True),
    "groq_deepseek": ModelProfile("groq_deepseek", "groq", os.getenv("GROQ_DEEPSEEK_MODEL", "deepseek-r1-distill-llama-70b"), "GROQ_API_KEY", float(os.getenv("GROQ_DEEPSEEK_INPUT_PER_M", "0.75")), float(os.getenv("GROQ_DEEPSEEK_OUTPUT_PER_M", "0.99"))),
    "groq_qwen": ModelProfile("groq_qwen", "groq", os.getenv("GROQ_QWEN_MODEL", "qwen/qwen3-32b"), "GROQ_API_KEY", float(os.getenv("GROQ_QWEN_INPUT_PER_M", "0.29")), float(os.getenv("GROQ_QWEN_OUTPUT_PER_M", "0.59"))),
}

def json_prompt(transcript: str, instruction: str) -> str:
    return f"{instruction}\nReturn exactly one JSON object with arrays named tasks and risks. Include only explicitly stated information; do not fabricate facts.\n\nTranscript:\n{transcript}"

def build_prompt(strategy: str, transcript: str) -> str:
    if strategy == "zero_shot": return json_prompt(transcript, "Extract all tasks and risks from this transcript into JSON.")
    if strategy == "few_shot": return f"Extract tasks and risks as JSON. Follow the examples exactly.\n\n{EXAMPLE_ONE}\n\n{EXAMPLE_TWO}\n\n{json_prompt(transcript, 'Now process this target transcript.') }"
    if strategy == "cot": return json_prompt(transcript, "First identify all speakers. Second identify all future actions. Third identify all explicitly stated risks. Perform that reasoning internally, then output only the final JSON.")
    if strategy == "xml": return f"Extract only explicitly stated tasks and risks. Return only XML in this exact shape: <Meeting><Tasks><Task>text</Task></Tasks><Risks><Risk>text</Risk></Risks></Meeting>. Do not fabricate facts.\n\nTranscript:\n{transcript}"
    raise ValueError(f"Unknown strategy: {strategy}")

def parse_xml(text: str) -> dict[str, list[str]]:
    start, end = text.find("<Meeting"), text.rfind("</Meeting>")
    if start < 0 or end < start: raise ValueError("No <Meeting> XML document found")
    root = ET.fromstring(text[start:end + len("</Meeting>")])
    if root.tag != "Meeting": raise ValueError("XML root must be Meeting")
    tasks = [(node.text or "").strip() for node in root.findall("./Tasks/Task") if (node.text or "").strip()]
    risks = [(node.text or "").strip() for node in root.findall("./Risks/Risk") if (node.text or "").strip()]
    return {"tasks": tasks, "risks": risks}

def parse_output(strategy: str, text: str) -> dict[str, list[str]]:
    parsed = parse_xml(text) if strategy == "xml" else extract_json(text)
    return {"tasks": string_list(parsed.get("tasks", []), "tasks"), "risks": string_list(parsed.get("risks", []), "risks")}

def context_stress_cases() -> list[dict[str, Any]]:
    filler = "The team reviewed historical release notes, telemetry trends, ownership changes, and routine operational details without making commitments."
    records = []
    for number, owner in enumerate(("Aniket", "Priya", "Luis", "Chen", "Fatima"), 1):
        transcript = f"Maya: {owner} will submit the incident recovery plan by Friday.\nMaya: Missing the plan could delay the compliance launch.\n" + "\n".join(f"Speaker {i}: {filler}" for i in range(1, 78))
        records.append({"id": f"context_stress_{number:03d}", "transcript": transcript, "ground_truth": {"tasks": [f"Submit the incident recovery plan by Friday (Assignee: {owner})"], "risks": ["Compliance launch delay caused by a missing incident recovery plan"]}, "tags": ["context_stress", "target_at_beginning"]})
    return records

def http_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: int = 90) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", **headers}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response: return json.loads(response.read().decode("utf-8")), (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error

def call_provider(profile: ModelProfile, prompt: str) -> tuple[str, dict[str, int | None], float, dict[str, Any]]:
    key = os.environ.get(profile.api_key_env)
    if not key: raise RuntimeError(f"Missing {profile.api_key_env}")
    if profile.provider in {"openai", "groq"}:
        url = "https://api.openai.com/v1/chat/completions" if profile.provider == "openai" else "https://api.groq.com/openai/v1/chat/completions"
        payload, latency = http_json(url, {"model": profile.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}, {"Authorization": f"Bearer {key}"})
        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        return text, {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens")}, latency, payload
    if profile.provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{profile.model}:generateContent?key={key}"
        payload, latency = http_json(url, {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}}, {})
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        usage = payload.get("usageMetadata", {})
        return text, {"prompt_tokens": usage.get("promptTokenCount"), "completion_tokens": usage.get("candidatesTokenCount"), "total_tokens": usage.get("totalTokenCount")}, latency, payload
    if profile.provider == "anthropic":
        payload, latency = http_json("https://api.anthropic.com/v1/messages", {"model": profile.model, "max_tokens": 1024, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}, {"x-api-key": key, "anthropic-version": "2023-06-01"})
        text = "".join(block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text")
        usage = payload.get("usage", {})
        return text, {"prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens"), "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)}, latency, payload
    raise ValueError(f"Unsupported provider: {profile.provider}")

def cost(profile: ModelProfile, usage: dict[str, int | None]) -> float | None:
    incoming, outgoing = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if incoming is None or outgoing is None: return None
    return round(incoming * profile.input_per_million_usd / 1_000_000 + outgoing * profile.output_per_million_usd / 1_000_000, 8)

# MODIFIED: Added judge_profile parameter to execute Task 10 automatically
async def run_case(profile: ModelProfile, strategy: str, row: dict[str, Any], log_file: Any, run_id: str, judge_profile: ModelProfile | None = None) -> Prediction:
    prompt = build_prompt(strategy, row["transcript"])
    timestamp = datetime.now(timezone.utc).isoformat()
    response, usage, latency, raw, parsed, error, parse_ms, judge_fabricated = "", {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}, None, None, {"tasks": [], "risks": []}, None, None, None
    try:
        response, usage, latency, raw = await asyncio.to_thread(call_provider, profile, prompt)
        started = time.perf_counter(); parsed = parse_output(strategy, response); parse_ms = round((time.perf_counter() - started) * 1000, 3)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        
    # TASK 10: Automatic Hallucination Judge Execution
    if judge_profile and not error and (parsed["tasks"] or parsed["risks"]):
        try:
            evidence = json.dumps({"tasks": parsed["tasks"], "risks": parsed["risks"]}, ensure_ascii=False)
            judge_prompt = JUDGE_PROMPT.format(transcript=row["transcript"], extracted_json=evidence)
            judge_raw, _, _, _ = await asyncio.to_thread(call_provider, judge_profile, judge_prompt)
            judge_fabricated = judge_count(judge_raw)
        except Exception as exc:
            error = f"Judge Evaluation Error: {exc}"

    record = {"timestamp_utc": timestamp, "run_id": run_id, "case_id": row["id"], "tags": row.get("tags", []), "strategy": strategy, "model_key": profile.key, "model": profile.model, "provider": profile.provider, "prompt": prompt, "response": response, "provider_response": raw, "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"], "total_tokens": usage["total_tokens"], "request_latency_ms": round(latency, 3) if latency is not None else None, "parse_latency_ms": parse_ms, "parse_error": error, "cost_usd": cost(profile, usage), "judge_fabricated_items": judge_fabricated}
    log_file.write(json.dumps(record, ensure_ascii=False) + "\n"); log_file.flush()
    if profile.native_rate_delay: await asyncio.sleep(2)
    
    return Prediction(row["id"], parsed["tasks"], parsed["risks"], latency, error, judge_fabricated, raw_response=response)

def aggregate(dataset: list[dict[str, Any]], results: dict[tuple[str, str], list[Prediction]], profiles: dict[str, ModelProfile], audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = []
    for (model_key, strategy), predictions in results.items():
        report = evaluate(dataset, predictions)
        stress_rows = [row for row in dataset if "context_stress" in row.get("tags", [])]
        stress_predictions = [item for item in predictions if item.id in {row["id"] for row in stress_rows}]
        stress_report = evaluate(stress_rows, stress_predictions) if stress_rows else None
        calls = [row for row in audit_rows if row["model_key"] == model_key and row["strategy"] == strategy]
        token_input = sum(row["prompt_tokens"] or 0 for row in calls)
        token_output = sum(row["completion_tokens"] or 0 for row in calls)
        total_cost = round(sum(row["cost_usd"] or 0.0 for row in calls), 8)
        parse_latencies = [row["parse_latency_ms"] for row in calls if row["parse_latency_ms"] is not None]
        groups.append({"model_key": model_key, "model": profiles[model_key].model, "provider": profiles[model_key].provider, "strategy": strategy, "evaluation": {key: value for key, value in report.items() if key != "per_record"}, "context_stress_evaluation": {key: value for key, value in stress_report.items() if key != "per_record"} if stress_report else None, "token_usage": {"prompt_tokens": token_input, "completion_tokens": token_output, "total_tokens": token_input + token_output}, "cost_usd": total_cost, "xml_or_json_parse_latency_ms": round(sum(parse_latencies) / len(parse_latencies), 3) if parse_latencies else None, "context_stress_cases": len(stress_rows)})
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "records": len(dataset), "groups": groups, "pricing": {key: {"input_per_million_usd": profile.input_per_million_usd, "output_per_million_usd": profile.output_per_million_usd} for key, profile in profiles.items()}}

# ADDED: Automated Markdown Generator to separate results by Task natively
def generate_markdown_reports(summary: dict[str, Any], output_dir: Path) -> None:
    groups = summary.get("groups", [])
    
    # Task 1: Basic Evaluation Details
    with (output_dir / "Task_1_Evaluation.md").open("w", encoding="utf-8") as f:
        f.write("# Task 1: Baseline Evaluation Architecture\n\n")
        f.write(f"**Total Records Evaluated:** {summary.get('records', 0)}\n\n")
        f.write("Pipeline successfully extracted structured metrics (Precision, Recall, F1) across the benchmark dataset using Ground Truth comparison.\n")

    # Task 2: Prompt Engineering Case Study (Averaged by Strategy)
    strategies = {}
    for g in groups:
        s = g["strategy"]
        if s not in strategies: strategies[s] = {"f1": [], "errors": [], "latency": []}
        strategies[s]["f1"].append(g["evaluation"]["item_metrics"]["combined"]["f1_score"])
        strategies[s]["errors"].append(g["evaluation"]["operational_metrics"]["json_parse_error_rate"])
        if g["evaluation"]["operational_metrics"]["average_latency_ms"]:
            strategies[s]["latency"].append(g["evaluation"]["operational_metrics"]["average_latency_ms"])
            
    with (output_dir / "Task_2_Prompt_Engineering.md").open("w", encoding="utf-8") as f:
        f.write("# Task 2: Prompt Engineering Comparison\n\n")
        f.write("| Prompt Strategy | Avg F1 Score | Parse Error Rate | Avg Request Latency (ms) |\n")
        f.write("|---|---|---|---|\n")
        for s, data in strategies.items():
            avg_f1 = sum(data["f1"])/len(data["f1"]) if data["f1"] else 0
            avg_err = sum(data["errors"])/len(data["errors"]) if data["errors"] else 0
            avg_lat = sum(data["latency"])/len(data["latency"]) if data["latency"] else 0
            f.write(f"| {s} | {avg_f1:.4f} | {avg_err:.4f} | {avg_lat:.2f} |\n")

    # Task 3 & 10: Model Comparison and Hallucinations
    with (output_dir / "Task_3_and_10_Model_Comparison.md").open("w", encoding="utf-8") as f:
        f.write("# Task 3 & 10: Model Showdown & Hallucination Defense\n\n")
        f.write("| Model | Prompt Strategy | F1 Score | Cost (USD) | Hallucination Rate |\n")
        f.write("|---|---|---|---|---|\n")
        for g in groups:
            f1 = g["evaluation"]["item_metrics"]["combined"]["f1_score"]
            cost = g["cost_usd"]
            halluc = g["evaluation"]["hallucination"].get("judge_hallucination_rate")
            halluc_str = f"{halluc:.4f}" if halluc is not None else "N/A"
            f.write(f"| {g['model_key']} | {g['strategy']} | {f1:.4f} | ${cost:.6f} | {halluc_str} |\n")

async def main_async(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset)
    if args.include_context_stress: dataset += context_stress_cases()
    if args.limit: dataset = dataset[:args.limit]
    selected = list(PROFILES) if "all" in args.models else args.models
    profiles = {name: PROFILES[name] for name in selected}
    
    # Task 10: Define the Judge Model (defaults to Qwen via Groq for fast, cheap judging)
    judge_profile = PROFILES.get(args.judge_model) if args.judge_model else None
    
    for name, profile in profiles.items():
        if not os.environ.get(profile.api_key_env): raise SystemExit(f"{name} needs environment variable {profile.api_key_env}")
    
    if judge_profile and not os.environ.get(judge_profile.api_key_env):
        raise SystemExit(f"Judge {args.judge_model} needs environment variable {judge_profile.api_key_env}")

    LOG_DIR.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path, summary_path = LOG_DIR / f"benchmark_{run_id}.jsonl", LOG_DIR / f"benchmark_{run_id}_summary.json"
    results: dict[tuple[str, str], list[Prediction]] = {}
    
    with log_path.open("w", encoding="utf-8") as log_file:
        for model_key, profile in profiles.items():
            for strategy in args.strategies:
                print(f"Running {model_key} / {strategy} on {len(dataset)} records")
                predictions = []
                # MODIFIED: Passes the judge_profile down to run_case
                for row in dataset: predictions.append(await run_case(profile, strategy, row, log_file, run_id, judge_profile))
                results[(model_key, strategy)] = predictions
                
    audit_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    summary = aggregate(dataset, results, profiles, audit_rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    # NEW: Automatically generate the Markdown report files
    generate_markdown_reports(summary, LOG_DIR)
    
    print(f"Raw audit log: {log_path}\nSummary JSON: {summary_path}")
    print(f"Markdown Reports generated in: {LOG_DIR}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--models", nargs="+", choices=["all", *PROFILES], default=["all"])
    parser.add_argument("--strategies", nargs="+", choices=["zero_shot", "few_shot", "cot", "xml"], default=["zero_shot", "few_shot", "cot", "xml"])
    # NEW ARGUMENT: Allows user to specify a judge model
    parser.add_argument("--judge-model", choices=list(PROFILES.keys()), default="groq_qwen", help="Model used for Task 10 hallucination detection")
    parser.add_argument("--limit", type=int, help="Use only the first N records; useful for a smoke test")
    parser.add_argument("--include-context-stress", action=argparse.BooleanOptionalAction, default=True, help="Add five ~1,000-word target-at-beginning cases")
    asyncio.run(main_async(parser.parse_args()))

if __name__ == "__main__": main()