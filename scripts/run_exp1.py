#!/usr/bin/env python
"""Experiment 1 full run: Clean / Dirty / ValidRepair / Dirty+FD × 2 models × 1200."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from clients.bailian import make_bailian_client
from clients.zhipu import make_zhipu_client
from evaluation.runner import build_run_manifest, make_request_id, run_requests
from prompts.templates import PromptBuilder
from utils.io_utils import iter_jsonl, write_json


def _client(provider: str):
    return make_zhipu_client() if provider == "zhipu" else make_bailian_client()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["zhipu", "bailian", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--concurrency", type=int, default=config.DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    providers = ["zhipu", "bailian"] if args.provider == "both" else [args.provider]
    questions = list(iter_jsonl(config.QA_DIR / "questions.jsonl"))
    if args.limit:
        questions = questions[: args.limit]
    evidence = {e["question_id"]: e for e in iter_jsonl(config.QA_DIR / "evidence.jsonl")}
    builder = PromptBuilder()

    for provider in providers:
        if not config.get_api_key(provider):
            print(f"API key detected: NO ({provider}) — skip")
            continue
        print(f"API key detected: YES ({provider})")
        model = config.MODELS[provider]["model"]
        client = _client(provider)
        jobs = []
        for q in questions:
            e = evidence[q["question_id"]]
            for condition, ev_key in (
                ("clean", "clean_evidence"),
                ("dirty", "dirty_evidence"),
                ("valid_repair", "valid_repair_evidence"),
            ):
                sys_p, user_p = builder.qa_prompts(q["question"], e[ev_key])
                jobs.append(
                    {
                        "request_id": make_request_id("exp1", provider, condition, q["question_id"]),
                        "experiment": "exp1",
                        "condition": condition,
                        "question_id": q["question_id"],
                        "question_category": q["category"],
                        "system_prompt": sys_p,
                        "user_prompt": user_p,
                        "ground_truth": q["answer"],
                        "model": model,
                        "dataset_version": config.DATASET_VERSION,
                    }
                )
            sys_p, user_p = builder.conflict_prompts(
                q["relation"], q["question"], e["dirty_evidence"]
            )
            expect = "CONFLICT" if q["category"] == "Answer-Critical-Conflict" else None
            jobs.append(
                {
                    "request_id": make_request_id("exp1", provider, "dirty_fd", q["question_id"]),
                    "experiment": "exp1",
                    "condition": "dirty_fd",
                    "question_id": q["question_id"],
                    "question_category": q["category"],
                    "system_prompt": sys_p,
                    "user_prompt": user_p,
                    "ground_truth": q["answer"],
                    "expect_special": expect,
                    "model": model,
                    "dataset_version": config.DATASET_VERSION,
                }
            )

        out_path = config.RAW_DIR / f"exp1_{provider}.jsonl"
        write_json(
            config.LOGS_DIR / f"exp1_{provider}_manifest.json",
            build_run_manifest(
                run_id=f"exp1_{provider}",
                provider=provider,
                model=model,
                experiment="exp1",
                question_ids=[q["question_id"] for q in questions],
                dataset_version=config.DATASET_VERSION,
            ),
        )
        print(f"[{provider}] model={model} jobs={len(jobs)}")
        run_requests(
            client,
            jobs,
            out_path,
            dry_run=args.dry_run,
            force=args.force,
            concurrency=args.concurrency,
        )


if __name__ == "__main__":
    main()
