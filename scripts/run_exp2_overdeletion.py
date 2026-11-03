#!/usr/bin/env python
"""Experiment 2A: OverDeleted vs CheckerCompleted (eligible unique-key protocol)."""

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
from generation.schemas import get_schema
from prompts.templates import PromptBuilder
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
    over = read_json(config.CANDIDATE_DIR / "overdeletion.json")
    manifest = read_json(config.CANDIDATE_DIR / "exp2a_question_manifest.json")
    builder = PromptBuilder()

    for provider in providers:
        if not config.get_api_key(provider):
            print(f"API key detected: NO ({provider}) — skip")
            continue
        print(f"API key detected: YES ({provider})")
        model = config.MODELS[provider]["model"]
        client = _client(provider)
        jobs = []
        for run in manifest["runs"]:
            rate = float(run["error_rate"])
            seed = int(run["corruption_seed"])
            rate_key = f"{rate:.2f}"
            art_root = over["error_rates"][rate_key][f"seed{seed}"]
            eval_ids = run["all_eval_question_ids"]
            affected = set(run["affected_question_ids"])
            for qid in eval_ids:
                q = questions[qid]
                rel_art = art_root[q["relation"]]
                for condition, cand_key in (
                    ("candidate", "initial_candidate"),
                    ("checked", "completed_candidate"),
                ):
                    ev = _evidence_from_candidate(q, rel_art[cand_key])
                    sys_p, user_p = builder.qa_prompts(q["question"], ev)
                    cond_tag = f"{rate_key}__{condition}__seed{seed}"
                    jobs.append(
                        {
                            "request_id": make_request_id(
                                "exp2a", provider, cond_tag, qid
                            ),
                            "experiment": "exp2a",
                            "condition": condition,
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
                            "checker_meta": {
                                "checker_iterations": rel_art.get("checker_iterations"),
                                "added_back_count": rel_art.get("added_back_count"),
                            },
                        }
                    )

        out_path = config.RAW_DIR / f"exp2_overdeletion_{provider}.jsonl"
        write_json(
            config.LOGS_DIR / f"exp2a_{provider}_manifest.json",
            build_run_manifest(
                run_id=f"exp2a_{provider}",
                provider=provider,
                model=model,
                experiment="exp2a",
                question_ids=sorted({j["question_id"] for j in jobs}),
                dataset_version=config.DATASET_VERSION,
            ),
        )
        print(f"[{provider}] jobs={len(jobs)}")
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
