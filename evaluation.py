"""Scores meeting extraction against ground truth and checks hallucinations."""
from __future__ import annotations
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

DATASET = Path(__file__).with_name("test_dataset.json")
BASE_URL = "https://api.groq.com/openai/v1"
EXTRACTION_PROMPT = """Extract only explicitly stated meeting tasks and risks. Return exactly one JSON object with arrays named tasks and risks. Do not invent names, deadlines, actions, or risks. Use empty arrays when absent.\n\nTranscript:\n{transcript}"""
JUDGE_PROMPT = """Given the Transcript and Extracted JSON, count task or risk items that contain information, names, or deadlines NOT explicitly stated in the transcript. Count an item once even if it has several fabricated facts. Return only the integer count.\n\nTranscript:\n{transcript}\n\nExtracted JSON:\n{extracted_json}"""
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
@dataclass
class Counts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    def add(self, other: "Counts") -> None:
        self.true_positive += other.true_positive
        self.false_positive += other.false_positive
        self.false_negative += other.false_negative
    def scores(self) -> dict[str, float]:
        tp, fp, fn = self.true_positive, self.false_positive, self.false_negative
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        return {"accuracy": round(accuracy, 4), "precision": round(precision, 4), "recall": round(recall, 4), "f1_score": round(f1, 4)}

@dataclass
class Prediction:
    id: str
    tasks: list[str]
    risks: list[str]
    latency_ms: float | None = None
    parse_error: str | None = None
    judge_fabricated_items: int | None = None
    raw_response: str | None = None

def string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a JSON array of strings")
    return value

def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Dataset must be a JSON array")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("Dataset entries require unique string ids")
        ids.add(row["id"])
        truth = row.get("ground_truth")
        if not isinstance(row.get("transcript"), str) or not isinstance(truth, dict):
            raise ValueError(f"Invalid record: {row['id']}")
        string_list(truth.get("tasks", []), "ground_truth.tasks")
        string_list(truth.get("risks", []), "ground_truth.risks")
    return rows

def to_prediction(row: dict[str, Any]) -> Prediction:
    return Prediction(str(row["id"]), string_list(row.get("tasks", []), "tasks"), string_list(row.get("risks", []), "risks"), row.get("latency_ms"), row.get("parse_error"), row.get("judge_fabricated_items"), row.get("raw_response"))

def load_predictions(path: Path) -> list[Prediction]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Predictions must be a JSON array")
    predictions = [to_prediction(row) for row in rows]
    if len({item.id for item in predictions}) != len(predictions):
        raise ValueError("Prediction ids must be unique")
    return predictions

def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()

def score_items(expected: Iterable[str], predicted: Iterable[str]) -> Counts:
    expected_values = [normalise(item) for item in expected]
    unmatched, result = set(range(len(expected_values))), Counts()
    for item in predicted:
        match = next((i for i in unmatched if expected_values[i] == normalise(item)), None)
        if match is None:
            result.false_positive += 1
        else:
            unmatched.remove(match)
            result.true_positive += 1
    result.false_negative = len(unmatched)
    return result

def evaluate(dataset: list[dict[str, Any]], predictions: list[Prediction]) -> dict[str, Any]:
    by_id = {item.id: item for item in predictions}
    unknown = set(by_id) - {row["id"] for row in dataset}
    if unknown:
        raise ValueError(f"Unknown prediction id(s): {', '.join(sorted(unknown)[:5])}")
    totals = {"tasks": Counts(), "risks": Counts(), "combined": Counts()}
    details, items, judge_total, judged, errors, latencies = [], 0, 0, 0, 0, []
    for row in dataset:
        pred = by_id.get(row["id"], Prediction(row["id"], [], []))
        task, risk, combined = score_items(row["ground_truth"]["tasks"], pred.tasks), score_items(row["ground_truth"]["risks"], pred.risks), Counts()
        combined.add(task); combined.add(risk)
        for name, count in (("tasks", task), ("risks", risk), ("combined", combined)): totals[name].add(count)
        items += len(pred.tasks) + len(pred.risks)
        if pred.judge_fabricated_items is not None: judge_total += pred.judge_fabricated_items; judged += 1
        if pred.latency_ms is not None: latencies.append(float(pred.latency_ms))
        errors += int(pred.parse_error is not None)
        details.append({"id": row["id"], "tasks": asdict(task), "risks": asdict(risk), "combined": asdict(combined), "parse_error": pred.parse_error, "judge_fabricated_items": pred.judge_fabricated_items})
    combined = totals["combined"]
    return {"records_in_dataset": len(dataset), "records_predicted": len(predictions), "item_metrics": {name: {"counts": asdict(count), **count.scores()} for name, count in totals.items()}, "hallucination": {"false_positives": combined.false_positive, "false_positive_rate": round(combined.false_positive / items, 4) if items else 0.0, "judge_records": judged, "judge_fabricated_items": judge_total if judged else None, "judge_hallucination_rate": round(judge_total / items, 4) if judged and items else None}, "operational_metrics": {"json_parse_errors": errors, "json_parse_error_rate": round(errors / len(dataset), 4) if dataset else 0.0, "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None}, "per_record": details}

def extract_json(text: str) -> dict[str, Any]:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try: value = json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}")
        if start < 0 or end < start: raise
        value = json.loads(clean[start:end + 1])
    if not isinstance(value, dict): raise ValueError("Model output is not a JSON object")
    return value

class ChatClient:
    def __init__(self, api_key: str, base_url: str, timeout: int = 60):
        self.api_key, self.url, self.timeout = api_key, base_url.rstrip("/") + "/chat/completions", timeout
    def chat(self, model: str, prompt: str) -> tuple[str, float]:
        request = urllib.request.Request(self.url, data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode(), headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response: payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"API request failed ({error.code}): {error.read().decode(errors='replace')}") from error
        try: content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error: raise RuntimeError(f"Unexpected API response: {payload}") from error
        if not isinstance(content, str): raise RuntimeError("API content was not text")
        return content, (time.perf_counter() - started) * 1000

def judge_count(text: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*", text)
    if not match: raise ValueError(f"Judge did not return only an integer: {text[:100]!r}")
    return int(match.group(1))

def run_model(dataset: list[dict[str, Any]], client: ChatClient, model: str, judge_model: str | None) -> list[Prediction]:
    output = []
    for number, row in enumerate(dataset, 1):
        print(f"Extracting {number}/{len(dataset)}: {row['id']}")
        raw, latency = client.chat(model, EXTRACTION_PROMPT.format(transcript=row["transcript"]))
        try:
            parsed = extract_json(raw)
            pred = Prediction(row["id"], string_list(parsed.get("tasks", []), "tasks"), string_list(parsed.get("risks", []), "risks"), round(latency, 2), raw_response=raw)
        except (ValueError, json.JSONDecodeError) as error: pred = Prediction(row["id"], [], [], round(latency, 2), parse_error=str(error), raw_response=raw)
        if judge_model:
            evidence = json.dumps({"tasks": pred.tasks, "risks": pred.risks}, ensure_ascii=False)
            judge_raw, _ = client.chat(judge_model, JUDGE_PROMPT.format(transcript=row["transcript"], extracted_json=evidence))
            try: pred.judge_fabricated_items = judge_count(judge_raw)
            except ValueError as error: pred.parse_error = f"{pred.parse_error}; {error}" if pred.parse_error else str(error)
        output.append(pred)
        if number < len(dataset):
            time.sleep(2)
    return output

def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    commands = parser.add_subparsers(dest="command", required=True)
    offline = commands.add_parser("evaluate", help="Score saved predictions without API calls")
    offline.add_argument("--predictions", type=Path, required=True); offline.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    perfect = commands.add_parser("make-perfect-predictions", help="Create no-API smoke-test predictions")
    perfect.add_argument("--output", type=Path, default=Path("perfect_predictions.json"))
    run = commands.add_parser("run", help="Run an extraction model and optional hallucination judge")
    run.add_argument("--model", required=True); run.add_argument("--judge-model")
    run.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", BASE_URL)); run.add_argument("--api-key", default=os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")); run.add_argument("--limit", type=int)
    run.add_argument("--predictions-output", type=Path, default=Path("model_predictions.json")); run.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    args = parser.parse_args(); dataset = load_dataset(args.dataset)
    if args.command == "make-perfect-predictions":
        write_json(args.output, [asdict(Prediction(row["id"], row["ground_truth"]["tasks"], row["ground_truth"]["risks"])) for row in dataset]); print(f"Wrote {len(dataset)} perfect predictions to {args.output}"); return
    if args.command == "evaluate": result = evaluate(dataset, load_predictions(args.predictions)); result["run_metadata"] = {"mode": "offline", "dataset": str(args.dataset)}; write_json(args.output, result)
    else:
        if not args.api_key: parser.error("--api-key or LLM_API_KEY is required for run")
        selected = dataset[:args.limit] if args.limit else dataset
        predictions = run_model(selected, ChatClient(args.api_key, args.base_url), args.model, args.judge_model)
        write_json(args.predictions_output, [asdict(item) for item in predictions]); result = evaluate(selected, predictions); result["run_metadata"] = {"mode": "api", "model": args.model, "judge_model": args.judge_model, "base_url": args.base_url, "dataset": str(args.dataset)}; write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "per_record"}, indent=2))

if __name__ == "__main__": main()
