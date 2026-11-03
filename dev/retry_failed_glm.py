#!/usr/bin/env python
"""Retry ONLY failed GLM formal requests; flush each success immediately."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from clients.openai_compatible import OpenAICompatibleClient
from evaluation.runner import evaluate_qa_row
from generation.schemas import get_schema
from prompts.templates import PROMPT_VERSION, PromptBuilder
from utils.io_utils import iter_jsonl, read_json, sha256_text, write_jsonl

BACKOFF = (2, 4, 8, 16, 32, 32, 32, 32, 32, 32)
MAX_RETRIES = 10

FILES = [
    ("exp1", config.RAW_DIR / "exp1_zhipu.jsonl", "exp1__"),
    ("exp2a", config.RAW_DIR / "exp2_overdeletion_zhipu.jsonl", "exp2a__"),
    ("exp2b", config.RAW_DIR / "exp2_residual_zhipu.jsonl", "exp2b__"),
]


def _load_qa():
    questions = {q["question_id"]: q for q in iter_jsonl(config.QA_DIR / "questions.jsonl")}
    evidence = {e["question_id"]: e for e in iter_jsonl(config.QA_DIR / "evidence.jsonl")}
    return questions, evidence


def _evidence_from_candidate(question: dict, candidate_rows: list[dict]) -> list[dict]:
    schema = get_schema(question["relation"])
    key = question["query_key"]
    matched = [r for r in candidate_rows if all(r.get(a) == key.get(a) for a in key)]
    distractors = []
    for r in candidate_rows:
        if all(r.get(a) == key.get(a) for a in key):
            continue
        distractors.append(r)
        if len(distractors) >= 2:
            break
    return [{a: rec[a] for a in schema.attributes} for rec in (matched + distractors)]


def rebuild_job(failed_row: dict, questions: dict, evidence: dict, over: dict, residual: dict) -> dict:
    rid = failed_row["request_id"]
    builder = PromptBuilder()
    parts = rid.split("__")
    if parts[0] == "exp1":
        condition, qid = parts[2], parts[3]
        q = questions[qid]
        e = evidence[qid]
        if condition == "dirty_fd":
            sys_p, user_p = builder.conflict_prompts(q["relation"], q["question"], e["dirty_evidence"])
            expect = "CONFLICT" if q["category"] == "Answer-Critical-Conflict" else None
        else:
            ev_key = {
                "clean": "clean_evidence",
                "dirty": "dirty_evidence",
                "valid_repair": "valid_repair_evidence",
            }[condition]
            sys_p, user_p = builder.qa_prompts(q["question"], e[ev_key])
            expect = None
        return {
            "request_id": rid,
            "experiment": "exp1",
            "condition": condition,
            "question_id": qid,
            "question_category": q["category"],
            "system_prompt": sys_p,
            "user_prompt": user_p,
            "ground_truth": q["answer"],
            "expect_special": expect,
            "dataset_version": config.DATASET_VERSION,
            "model": config.MODELS["zhipu"]["model"],
        }
    if parts[0] == "exp2a":
        rate_s, condition, seed_s, qid = parts[2], parts[3], parts[4], parts[5]
        seed = int(seed_s.replace("seed", ""))
        rate = float(rate_s)
        q = questions[qid]
        art = over["error_rates"][f"{rate:.2f}"][f"seed{seed}"][q["relation"]]
        cand_key = "completed_candidate" if condition == "checked" else "initial_candidate"
        ev = _evidence_from_candidate(q, art[cand_key])
        sys_p, user_p = builder.qa_prompts(q["question"], ev)
        return {
            "request_id": rid,
            "experiment": "exp2a",
            "condition": condition,
            "error_rate": rate,
            "corruption_seed": seed,
            "question_id": qid,
            "question_category": q["category"],
            "subset": failed_row.get("subset"),
            "system_prompt": sys_p,
            "user_prompt": user_p,
            "ground_truth": q["answer"],
            "dataset_version": config.DATASET_VERSION,
            "model": config.MODELS["zhipu"]["model"],
            "checker_meta": {
                "checker_iterations": art.get("checker_iterations"),
                "added_back_count": art.get("added_back_count"),
            },
        }
    if parts[0] == "exp2b":
        rate_s, condition, seed_s, qid = parts[2], parts[3], parts[4], parts[5]
        if condition != "unchecked":
            raise ValueError(f"Gate should not retry via LLM: {rid}")
        seed = int(seed_s.replace("seed", ""))
        rate = float(rate_s)
        q = questions[qid]
        art = residual["error_rates"][f"{rate:.2f}"][f"seed{seed}"][q["relation"]]
        ev = _evidence_from_candidate(q, art["candidate"])
        sys_p, user_p = builder.qa_prompts(q["question"], ev)
        return {
            "request_id": rid,
            "experiment": "exp2b",
            "condition": "unchecked",
            "error_rate": rate,
            "corruption_seed": seed,
            "question_id": qid,
            "question_category": q["category"],
            "subset": failed_row.get("subset"),
            "system_prompt": sys_p,
            "user_prompt": user_p,
            "ground_truth": q["answer"],
            "dataset_version": config.DATASET_VERSION,
            "model": config.MODELS["zhipu"]["model"],
        }
    raise ValueError(f"Unrecognized request_id: {rid}")


def run_one(client: OpenAICompatibleClient, job: dict) -> dict:
    resp = client.generate(job["system_prompt"], job["user_prompt"], job["request_id"])
    eval_part = evaluate_qa_row(
        resp.raw_text, job.get("ground_truth"), expect_special=job.get("expect_special")
    )
    row = {
        "request_id": job["request_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": job.get("experiment"),
        "condition": job.get("condition"),
        "question_id": job.get("question_id"),
        "question_category": job.get("question_category"),
        "provider": "zhipu",
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
        "retry_fill": True,
    }
    for k in ("error_rate", "corruption_seed", "subset", "checker_meta"):
        if k in job and job[k] is not None:
            row[k] = job[k]
    return row


def replace_in_file(path: Path, new_row: dict) -> None:
    """Replace the failed row with new_row; never modify other successful rows."""
    rid = new_row["request_id"]
    rows = list(iter_jsonl(path))
    out = []
    replaced = False
    for r in rows:
        if r.get("request_id") == rid and not r.get("success"):
            if new_row.get("success"):
                out.append(new_row)
                replaced = True
            else:
                out.append(new_row)  # update failure info
                replaced = True
        else:
            out.append(r)
    if not replaced and new_row.get("success"):
        # success already present — do nothing
        return
    write_jsonl(path, out)


def collect_failures(path: Path) -> list[dict]:
    rows = list(iter_jsonl(path))
    succ = {r["request_id"] for r in rows if r.get("success")}
    return [r for r in rows if (not r.get("success")) and r["request_id"] not in succ]


def main() -> int:
    log_path = config.LOGS_DIR / "retry_glm.log"
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)

    print("API key detected:", "YES" if config.get_api_key("zhipu") else "NO", flush=True)
    questions, evidence = _load_qa()
    over = read_json(config.CANDIDATE_DIR / "overdeletion.json")
    residual = read_json(config.CANDIDATE_DIR / "residual.json")

    pending: list[tuple[Path, dict]] = []
    for label, path, _prefix in FILES:
        fails = collect_failures(path)
        print(f"{label}: failures_to_retry={len(fails)}", flush=True)
        for fr in fails:
            pending.append((path, fr))

    print(f"TOTAL_TO_RETRY={len(pending)}", flush=True)
    if not pending:
        print("Nothing to retry.", flush=True)
        return 0

    meta = config.MODELS["zhipu"]
    client = OpenAICompatibleClient(
        provider="zhipu",
        model=meta["model"],
        base_url=meta["base_url"],
        api_key=config.get_api_key("zhipu"),
        max_retries=MAX_RETRIES,
        retry_backoff_sec=BACKOFF,
    )

    permanent = []
    for i, (path, fr) in enumerate(pending, 1):
        # re-check in case already filled by prior partial run
        if fr["request_id"] not in {r["request_id"] for r in collect_failures(path)}:
            print(f"[{i}/{len(pending)}] SKIP already filled {fr['request_id']}", flush=True)
            continue
        job = rebuild_job(fr, questions, evidence, over, residual)
        print(f"[{i}/{len(pending)}] retry {job['request_id']}", flush=True)
        if i > 1:
            time.sleep(1.5)
        row = run_one(client, job)
        replace_in_file(path, row)
        if row.get("success"):
            print(f"  OK attempts={row['attempt_count']} flushed", flush=True)
        else:
            print(f"  FAIL error={row.get('error')}", flush=True)
            permanent.append((job["request_id"], row.get("error")))

    # Final integrity
    print("\nFINAL_COUNTS:", flush=True)
    all_ok = True
    for label, path, _ in FILES:
        rows = list(iter_jsonl(path))
        # unique request_ids
        by_id = {}
        for r in rows:
            rid = r["request_id"]
            if rid not in by_id or (r.get("success") and not by_id[rid].get("success")):
                by_id[rid] = r
        n = len(by_id)
        ok = sum(1 for r in by_id.values() if r.get("success"))
        print(f"  {label}: {ok}/{n}", flush=True)
        expected = {"exp1": 4800, "exp2a": 3136, "exp2b": 2072}[label]
        if ok != expected or n != expected:
            all_ok = False

    if permanent:
        print("PERMANENT_FAILURES:", flush=True)
        for rid, err in permanent:
            print(f"  {rid}: {err}", flush=True)
        return 2
    print("ALL_FILLED_OK" if all_ok else "PARTIAL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
