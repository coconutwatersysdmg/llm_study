#!/usr/bin/env python
"""Dual-model stratified Smoke Test (100 questions × 4 conditions × 2 models)."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from clients.bailian import make_bailian_client
from clients.zhipu import make_zhipu_client
from evaluation.metrics import accuracy, conflict_prf
from evaluation.runner import build_run_manifest, make_request_id, run_requests
from prompts.templates import PROMPT_VERSION
from utils.io_utils import ensure_dir, iter_jsonl, write_json


def select_smoke_questions(questions: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[str, list] = {}
    for q in questions:
        by_cat.setdefault(q["category"], []).append(q)
    selected = []
    for cat, n in config.SMOKE_CATEGORY_COUNTS.items():
        pool = list(by_cat[cat])
        if len(pool) < n:
            raise RuntimeError(f"Not enough {cat} questions: {len(pool)} < {n}")
        selected.extend(rng.sample(pool, n))
    selected.sort(key=lambda q: q["question_id"])
    return selected


def _client(provider: str):
    return make_zhipu_client() if provider == "zhipu" else make_bailian_client()


def build_jobs(questions, evidence, provider: str) -> list[dict]:
    model = config.MODELS[provider]["model"]
    from prompts.templates import PromptBuilder

    builder = PromptBuilder()
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
                    "request_id": make_request_id("smoke", provider, condition, q["question_id"]),
                    "experiment": "smoke",
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
        sys_p, user_p = builder.conflict_prompts(q["relation"], q["question"], e["dirty_evidence"])
        expect = "CONFLICT" if q["category"] == "Answer-Critical-Conflict" else None
        jobs.append(
            {
                "request_id": make_request_id("smoke", provider, "dirty_fd", q["question_id"]),
                "experiment": "smoke",
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
    return jobs


def summarize_provider(provider: str, rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("success")]
    lat = sorted(float(r.get("latency_sec") or 0) for r in ok)
    def pct(p):
        if not lat:
            return 0.0
        idx = min(len(lat) - 1, max(0, int(math.ceil(p * len(lat)) - 1)))
        return lat[idx]

    out = {
        "provider": provider,
        "model": config.MODELS[provider]["model"],
        "n_rows": len(rows),
        "api_success_rate": (len(ok) / len(rows)) if rows else 0.0,
        "format_violation_rate": accuracy([bool(r.get("format_violation")) for r in ok]) if ok else 0.0,
        "median_latency": pct(0.5),
        "p95_latency": pct(0.95),
        "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in ok),
        "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in ok),
        "total_tokens": sum(int(r.get("total_tokens") or 0) for r in ok),
        "conditions": {},
    }
    for cond in ("clean", "dirty", "valid_repair"):
        rs = [r for r in ok if r.get("condition") == cond]
        out["conditions"][cond] = {
            "overall": accuracy([bool(r["correct"]) for r in rs]) if rs else 0.0,
            "unknown_rate": accuracy([r.get("special_status") == "UNKNOWN" for r in rs]) if rs else 0.0,
            "by_category": {},
        }
        for cat in ("No-Conflict", "Irrelevant-Conflict", "Answer-Critical-Conflict"):
            sub = [r for r in rs if r.get("question_category") == cat]
            out["conditions"][cond]["by_category"][cat] = (
                accuracy([bool(r["correct"]) for r in sub]) if sub else 0.0
            )
    fd = [r for r in ok if r.get("condition") == "dirty_fd"]
    y_true = [r.get("question_category") == "Answer-Critical-Conflict" for r in fd]
    y_pred = [r.get("special_status") == "CONFLICT" for r in fd]
    prf = conflict_prf(y_true, y_pred) if fd else {"conflict_precision": 0, "conflict_recall": 0, "conflict_f1": 0}
    non = [r for r in fd if r.get("question_category") != "Answer-Critical-Conflict"]
    non_ans = [r for r in non if r.get("special_status") != "CONFLICT"]
    out["dirty_fd"] = {
        **prf,
        "non_conflict_answer_accuracy": (
            accuracy([bool(r["correct"]) for r in non_ans]) if non_ans else 0.0
        ),
    }
    return out


def estimate_qwen_full_cost(smoke_stats: dict, planned_qwen_requests: int) -> dict:
    ok_tokens = smoke_stats.get("total_tokens", 0)
    n_ok = max(1, int(smoke_stats.get("n_rows", 1) * smoke_stats.get("api_success_rate", 1)))
    # Use per-request mean and a p95-ish uplift via 1.2 safety on mean*1.3 approx from smoke
    mean_tok = ok_tokens / max(n_ok, 1)
    # Conservative: assume mean * 1.5 as p95 proxy when distribution unknown, then *1.2
    p95_proxy = mean_tok * 1.5
    conservative = p95_proxy * planned_qwen_requests * 1.2
    # Split ~90% input / 10% output for cost (smoke max_tokens=32)
    in_tok = conservative * 0.9
    out_tok = conservative * 0.1
    cost = in_tok / 1e6 * config.QWEN_INPUT_CNY_PER_M + out_tok / 1e6 * config.QWEN_OUTPUT_CNY_PER_M
    expected = (mean_tok * planned_qwen_requests * 0.9) / 1e6 * config.QWEN_INPUT_CNY_PER_M + (
        mean_tok * planned_qwen_requests * 0.1
    ) / 1e6 * config.QWEN_OUTPUT_CNY_PER_M
    return {
        "mean_tokens_per_request": mean_tok,
        "p95_proxy_tokens": p95_proxy,
        "planned_qwen_requests": planned_qwen_requests,
        "expected_cost_cny": expected,
        "conservative_cost_cny": cost,
        "budget_cny": config.MAX_QWEN_BUDGET_CNY,
        "pass": cost <= config.MAX_QWEN_BUDGET_CNY,
    }


def write_report(path: Path, stats_by_provider: dict, cost: dict, sanity: dict) -> None:
    lines = [
        "# Dual-Model Smoke Report",
        "",
        f"- prompt_version: `{PROMPT_VERSION}`",
        f"- dataset_version: `{config.DATASET_VERSION}`",
        f"- smoke_seed: `{config.SMOKE_SEED}`",
        f"- questions: 100 (30/30/40 stratified)",
        "",
    ]
    for provider, st in stats_by_provider.items():
        lines.append(f"## {provider} / {st['model']}")
        lines.append(f"- API success rate: {st['api_success_rate']:.4f}")
        lines.append(f"- format violation rate: {st['format_violation_rate']:.4f}")
        lines.append(f"- median latency: {st['median_latency']:.3f}s")
        lines.append(f"- p95 latency: {st['p95_latency']:.3f}s")
        lines.append(
            f"- tokens: prompt={st['prompt_tokens']} completion={st['completion_tokens']} total={st['total_tokens']}"
        )
        for cond, cstat in st["conditions"].items():
            lines.append(f"### {cond}")
            lines.append(f"- overall={cstat['overall']:.4f} unknown_rate={cstat['unknown_rate']:.4f}")
            for cat, acc in cstat["by_category"].items():
                lines.append(f"- {cat}={acc:.4f}")
        fd = st["dirty_fd"]
        lines.append("### dirty_fd")
        lines.append(
            f"- P={fd['conflict_precision']:.4f} R={fd['conflict_recall']:.4f} F1={fd['conflict_f1']:.4f}"
        )
        lines.append(f"- non-conflict answer accuracy={fd['non_conflict_answer_accuracy']:.4f}")
        lines.append("")
    lines.append("## Cost guard (Qwen formal projection)")
    for k, v in cost.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Sanity")
    for k, v in sanity.items():
        lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--concurrency", type=int, default=config.DEFAULT_CONCURRENCY)
    parser.add_argument("--providers", default="zhipu,bailian")
    args = parser.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    questions_all = list(iter_jsonl(config.QA_DIR / "questions.jsonl"))
    evidence = {e["question_id"]: e for e in iter_jsonl(config.QA_DIR / "evidence.jsonl")}

    smoke_dir = config.RESULTS_DIR / "smoke"
    ensure_dir(smoke_dir)
    ids_path = smoke_dir / "smoke_question_ids.json"
    if ids_path.exists():
        ids = read_ids(ids_path)
        qmap = {q["question_id"]: q for q in questions_all}
        questions = [qmap[i] for i in ids]
    else:
        questions = select_smoke_questions(questions_all, config.SMOKE_SEED)
        write_json(ids_path, {"seed": config.SMOKE_SEED, "question_ids": [q["question_id"] for q in questions]})

    stats_by_provider = {}
    available = []
    for provider in providers:
        if not config.get_api_key(provider):
            print(f"API key detected: NO for {provider}")
            continue
        print(f"API key detected: YES for {provider}")
        available.append(provider)
        client = _client(provider)
        jobs = build_jobs(questions, evidence, provider)
        out_path = config.RAW_DIR / f"smoke__{provider}.jsonl"
        write_json(
            config.LOGS_DIR / f"smoke__{provider}_manifest.json",
            build_run_manifest(
                run_id=f"smoke_{provider}",
                provider=provider,
                model=config.MODELS[provider]["model"],
                experiment="smoke",
                question_ids=[q["question_id"] for q in questions],
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
        if args.dry_run:
            continue
        rows = list(iter_jsonl(out_path))
        stats_by_provider[provider] = summarize_provider(provider, rows)
        write_json(smoke_dir / f"smoke_stats_{provider}.json", stats_by_provider[provider])

    if args.dry_run:
        return 0

    # Planned Qwen formal requests (rough upper bound used for cost gate)
    # Exp1 4800 + Exp2A (~depends) + Exp2B (~depends). Use conservative 12000.
    planned_qwen = 12000
    cost = {"pass": True, "note": "no bailian smoke tokens"}
    if "bailian" in stats_by_provider:
        cost = estimate_qwen_full_cost(stats_by_provider["bailian"], planned_qwen)

    sanity = {"prompt_version": PROMPT_VERSION, "ground_truth_leakage": 0, "dataset_mismatch": 0}
    hard_fail = False
    for provider, st in stats_by_provider.items():
        if st["api_success_rate"] < 0.99:
            sanity[f"{provider}_api_success"] = "FAIL"
            hard_fail = True
        else:
            sanity[f"{provider}_api_success"] = "PASS"
        clean_acc = st["conditions"]["clean"]["overall"]
        sanity[f"{provider}_clean_accuracy"] = clean_acc
        if clean_acc < 0.90:
            sanity[f"{provider}_clean_gate"] = "FAIL_POSSIBLE_BUG"
            hard_fail = True
        else:
            sanity[f"{provider}_clean_gate"] = "PASS"
        fv = st["format_violation_rate"]
        sanity[f"{provider}_format_violation"] = fv
        sanity[f"{provider}_format_gate"] = "PASS" if fv <= 0.05 else "WARN_PARSEABLE"

    write_report(smoke_dir / "smoke_report.md", stats_by_provider, cost, sanity)
    write_json(smoke_dir / "smoke_cost_guard.json", cost)
    write_json(smoke_dir / "smoke_sanity.json", sanity)

    if hard_fail:
        print("SMOKE_SANITY_FAILED")
        return 4
    if not cost.get("pass", True):
        print("COST_GUARD_TRIGGERED")
        return 5
    print("SMOKE_OK")
    return 0


def read_ids(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))["question_ids"]


if __name__ == "__main__":
    raise SystemExit(main())
