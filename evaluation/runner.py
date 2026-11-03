"""Shared experiment runner: resume / cache / dry-run / concurrency."""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from clients.base import LLMClient, LLMResponse
from config import MODELS, PROJECT_ROOT, RAW_DIR
from evaluation.normalize import answers_equal, classify_special, normalize_answer
from prompts.templates import PROMPT_VERSION, PromptBuilder
from utils.io_utils import append_jsonl, ensure_dir, iter_jsonl, sha256_text, write_json


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT.parent / "experiments"),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def make_request_id(experiment: str, model_tag: str, condition: str, question_id: str) -> str:
    return f"{experiment}__{model_tag}__{condition}__{question_id}"


def load_success_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in iter_jsonl(path):
        if row.get("success") and row.get("request_id"):
            done.add(row["request_id"])
    return done


def estimate_tokens(text: str) -> int:
    # Rough multilingual estimate: ~2 chars / token for mixed CN/EN.
    return max(1, (len(text) + 1) // 2)


def build_run_manifest(
    run_id: str,
    provider: str,
    model: str,
    experiment: str,
    question_ids: list[str],
    dataset_version: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    meta = MODELS[provider]
    manifest = {
        "run_id": run_id,
        "date": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "model_snapshot": model,
        "provider": provider,
        "base_url": meta["base_url"],
        "temperature": 0.01,
        "max_tokens": 32,
        "thinking_mode": meta.get("enable_thinking"),
        "prompt_version": PROMPT_VERSION,
        "dataset_version": dataset_version,
        "question_ids": question_ids,
        "git_commit_hash": git_commit_hash(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def evaluate_qa_row(
    raw_text: str,
    ground_truth: Any,
    expect_special: Optional[str] = None,
) -> dict[str, Any]:
    normalized, format_violation = normalize_answer(raw_text)
    special = classify_special(normalized)
    if expect_special:
        correct = special == expect_special
    else:
        correct = answers_equal(normalized, ground_truth) if special is None else False
    return {
        "normalized_response": normalized,
        "special_status": special,
        "format_violation": format_violation,
        "correct": correct,
        "ground_truth": ground_truth,
    }


class ResultWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir(path.parent)

    def write(self, row: dict[str, Any]) -> None:
        append_jsonl(self.path, row)


def run_requests(
    client: LLMClient,
    jobs: list[dict[str, Any]],
    out_path: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    concurrency: int = 4,
    on_result: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Execute LLM jobs with resume + optional dry-run.

    Each job must contain:
      request_id, system_prompt, user_prompt, and any metadata to persist.
    """
    done = set() if force else load_success_ids(out_path)
    pending = [j for j in jobs if j["request_id"] not in done]
    stats = {
        "planned": len(jobs),
        "already_done": len(jobs) - len(pending),
        "to_run": len(pending),
        "dry_run": dry_run,
    }

    if dry_run:
        total_in = 0
        for j in pending:
            total_in += estimate_tokens(j["system_prompt"] + j["user_prompt"])
        stats["estimated_input_tokens"] = total_in
        stats["maximum_output_tokens"] = len(pending) * 32
        stats["models"] = sorted({j.get("model", client.model) for j in pending}) or [client.model]
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return stats

    writer = ResultWriter(out_path)
    builder = PromptBuilder()

    def _one(job: dict[str, Any]) -> dict[str, Any]:
        builder.assert_no_leakage(job["system_prompt"], job["user_prompt"])
        resp: LLMResponse = client.generate(
            job["system_prompt"], job["user_prompt"], job["request_id"]
        )
        eval_part = evaluate_qa_row(
            resp.raw_text,
            job.get("ground_truth"),
            expect_special=job.get("expect_special"),
        )
        row = {
            "request_id": job["request_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": job.get("experiment"),
            "condition": job.get("condition"),
            "question_id": job.get("question_id"),
            "question_category": job.get("question_category"),
            "provider": resp.provider,
            "model": resp.model,
            "prompt_version": PROMPT_VERSION,
            "dataset_version": job.get("dataset_version"),
            "system_prompt_hash": sha256_text(job["system_prompt"]),
            "user_prompt_hash": sha256_text(job["user_prompt"]),
            "prompt_hash": sha256_text(job["system_prompt"] + "\n" + job["user_prompt"]),
            "raw_response": resp.raw_text,
            "normalized_response": eval_part["normalized_response"],
            "ground_truth": eval_part["ground_truth"],
            "correct": eval_part["correct"],
            "special_status": eval_part["special_status"],
            "format_violation": eval_part["format_violation"],
            "latency_sec": resp.latency_sec,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "total_tokens": resp.total_tokens,
            "attempt_count": resp.attempts,
            "success": resp.success,
            "error": resp.error,
        }
        for k in (
            "error_rate",
            "corruption_seed",
            "subset",
            "gate_status",
            "checker_meta",
        ):
            if k in job:
                row[k] = job[k]
        return row

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(_one, j): j for j in pending}
        for fut in as_completed(futs):
            row = fut.result()
            writer.write(row)
            results.append(row)
            if on_result:
                on_result(row)

    stats["completed"] = len(results)
    stats["success"] = sum(1 for r in results if r.get("success"))
    return stats


def cache_key(provider: str, model: str, system_hash: str, user_hash: str) -> str:
    raw = f"{provider}|{model}|{system_hash}|{user_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
