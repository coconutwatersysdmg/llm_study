#!/usr/bin/env python
"""Experiment 2B: Residual conflict — Unchecked vs Checked Gate."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from clients.bailian import make_bailian_client
from clients.zhipu import make_zhipu_client
from evaluation.runner import (
    ResultWriter,
    build_run_manifest,
    load_success_ids,
    make_request_id,
    run_requests,
)
from generation.schemas import get_schema
from prompts.templates import PROMPT_VERSION, PromptBuilder
from utils.io_utils import iter_jsonl, read_json, write_json


def _client(provider: str):
    return make_zhipu_client() if provider == "zhipu" else make_bailian_client()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["zhipu", "bailian", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--concurrency", type=int, default=config.DEFAULT_CONCURRENCY)
    args = parser.parse_args()

    providers = ["zhipu", "bailian"] if args.provider == "both" else [args.provider]
    questions = {q["question_id"]: q for q in iter_jsonl(config.QA_DIR / "questions.jsonl")}
    residual = read_json(config.CANDIDATE_DIR / "residual.json")
    manifest = read_json(config.CANDIDATE_DIR / "exp2b_question_manifest.json")
    builder = PromptBuilder()

    for provider in providers:
        if not config.get_api_key(provider):
            print(f"API key detected: NO ({provider}) — skip")
            continue
        print(f"API key detected: YES ({provider})")
        model = config.MODELS[provider]["model"]
        client = _client(provider)
        out_path = config.RAW_DIR / f"exp2_residual_{provider}.jsonl"
        writer = ResultWriter(out_path)
        done = set() if args.force else load_success_ids(out_path)

        # Detection audit
        detection_ok = detection_total = 0
        for rate_key, seeds in residual["error_rates"].items():
            for seed_key, rels in seeds.items():
                for rel, art in rels.items():
                    detection_total += 1
                    if art.get("candidate_consistent") is False:
                        detection_ok += 1
        det_rate = detection_ok / detection_total if detection_total else 0.0
        if det_rate < 1.0:
            raise RuntimeError(
                f"RESIDUAL_CONFLICT_CHECKER_FAILURE detection_rate={det_rate}"
            )

        unchecked_jobs = []
        gated = 0
        for run in manifest["runs"]:
            rate = float(run["error_rate"])
            seed = int(run["corruption_seed"])
            rate_key = f"{rate:.2f}"
            art_root = residual["error_rates"][rate_key][f"seed{seed}"]
            affected = set(run["affected_question_ids"])
            for qid in run["all_eval_question_ids"]:
                q = questions[qid]
                art = art_root[q["relation"]]
                assert art["candidate_consistent"] is False

                gate_id = make_request_id(
                    "exp2b", provider, f"{rate_key}__checked_gate__seed{seed}", qid
                )
                if gate_id not in done and not args.dry_run:
                    writer.write(
                        {
                            "request_id": gate_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "experiment": "exp2b",
                            "condition": "checked_gate",
                            "error_rate": rate,
                            "corruption_seed": seed,
                            "question_id": qid,
                            "question_category": q["category"],
                            "subset": "affected" if qid in affected else "unaffected",
                            "provider": provider,
                            "model": model,
                            "prompt_version": PROMPT_VERSION,
                            "dataset_version": config.DATASET_VERSION,
                            "system_prompt_hash": None,
                            "user_prompt_hash": None,
                            "raw_response": "REJECTED_INVALID_CANDIDATE",
                            "normalized_response": "REJECTED_INVALID_CANDIDATE",
                            "ground_truth": q["answer"],
                            "correct": None,
                            "special_status": "REJECTED_INVALID_CANDIDATE",
                            "format_violation": False,
                            "latency_sec": 0.0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "attempt_count": 0,
                            "success": True,
                            "error": None,
                            "gate_status": "checker_rejected",
                            "checker_meta": {
                                "candidate_consistent": False,
                                "llm_called": False,
                            },
                        }
                    )
                    gated += 1
                elif args.dry_run:
                    gated += 1

                ev = _evidence_from_candidate(q, art["candidate"])
                sys_p, user_p = builder.qa_prompts(q["question"], ev)
                unchecked_jobs.append(
                    {
                        "request_id": make_request_id(
                            "exp2b",
                            provider,
                            f"{rate_key}__unchecked__seed{seed}",
                            qid,
                        ),
                        "experiment": "exp2b",
                        "condition": "unchecked",
                        "error_rate": rate,
                        "corruption_seed": seed,
                        "question_id": qid,
                        "question_category": q["category"],
                        "subset": "affected" if qid in affected else "unaffected",
                        "system_prompt": sys_p,
                        "user_prompt": user_p,
                        "ground_truth": q["answer"],
                        "model": model,
                        "dataset_version": config.DATASET_VERSION,
                    }
                )

        write_json(
            config.LOGS_DIR / f"exp2b_{provider}_manifest.json",
            build_run_manifest(
                run_id=f"exp2b_{provider}",
                provider=provider,
                model=model,
                experiment="exp2b",
                question_ids=sorted({j["question_id"] for j in unchecked_jobs}),
                dataset_version=config.DATASET_VERSION,
                extra={
                    "checker_detection_rate": det_rate,
                    "gated_count": gated,
                    "unchecked_llm_jobs": len(unchecked_jobs),
                },
            ),
        )
        print(
            f"[{provider}] detection_rate={det_rate} gated={gated} "
            f"unchecked_jobs={len(unchecked_jobs)}"
        )
        run_requests(
            client,
            unchecked_jobs,
            out_path,
            dry_run=args.dry_run,
            force=args.force,
            concurrency=args.concurrency,
        )


if __name__ == "__main__":
    main()
