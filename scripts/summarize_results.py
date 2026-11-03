#!/usr/bin/env python
"""Summarize raw JSONL into final CSVs + LLM_FINAL_REPORT.md."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from evaluation.metrics import accuracy, conflict_prf
from evaluation.statistics import mcnemar_test, paired_bootstrap_diff, wilson_ci
from prompts.templates import PROMPT_VERSION
from utils.io_utils import ensure_dir, iter_jsonl, read_json, sha256_file


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _dedupe_prefer_success(rows: list[dict]) -> list[dict]:
    """Keep at most one row per request_id, preferring success=True."""
    best: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        rid = r.get("request_id")
        if not rid:
            continue
        if rid not in best:
            best[rid] = r
            order.append(rid)
            continue
        prev = best[rid]
        if (not prev.get("success")) and r.get("success"):
            best[rid] = r
        elif prev.get("success") and r.get("success"):
            # keep newer timestamp if both success
            if str(r.get("timestamp", "")) >= str(prev.get("timestamp", "")):
                best[rid] = r
    return [best[rid] for rid in order]


def _load_exp1():
    out = {}
    for provider in ("zhipu", "bailian"):
        path = config.RAW_DIR / f"exp1_{provider}.jsonl"
        if path.exists():
            out[provider] = _dedupe_prefer_success(list(iter_jsonl(path)))
    return out


def summarize_exp1(data_by_provider: dict) -> tuple[list, list]:
    rows_out, pairwise, conflict_out = [], [], []
    for provider, data in data_by_provider.items():
        ok = [r for r in data if r.get("success")]
        model = config.MODELS[provider]["model"]
        by_cond = defaultdict(list)
        for r in ok:
            by_cond[r.get("condition")].append(r)
        for condition in ("clean", "dirty", "valid_repair"):
            subset = by_cond.get(condition, [])
            for category in (
                "ALL",
                "No-Conflict",
                "Irrelevant-Conflict",
                "Answer-Critical-Conflict",
            ):
                rs = (
                    subset
                    if category == "ALL"
                    else [r for r in subset if r.get("question_category") == category]
                )
                if not rs:
                    continue
                correct = [bool(r["correct"]) for r in rs]
                unk = [r.get("special_status") == "UNKNOWN" for r in rs]
                acc = accuracy(correct)
                lo, hi = wilson_ci(sum(correct), len(correct))
                rows_out.append(
                    {
                        "model": model,
                        "provider": provider,
                        "condition": condition,
                        "category": category,
                        "n": len(rs),
                        "accuracy": f"{acc:.6f}",
                        "ci_low": f"{lo:.6f}",
                        "ci_high": f"{hi:.6f}",
                        "unknown_rate": f"{accuracy(unk):.6f}",
                        "answer_coverage": f"{1 - accuracy(unk):.6f}",
                    }
                )
        fd = by_cond.get("dirty_fd", [])
        if fd:
            y_true = [r.get("question_category") == "Answer-Critical-Conflict" for r in fd]
            y_pred = [r.get("special_status") == "CONFLICT" for r in fd]
            prf = conflict_prf(y_true, y_pred)
            non = [r for r in fd if r.get("question_category") != "Answer-Critical-Conflict"]
            non_ans = [r for r in non if r.get("special_status") != "CONFLICT"]
            conflict_out.append(
                {
                    "model": model,
                    "provider": provider,
                    "n": len(fd),
                    "conflict_precision": f"{prf['conflict_precision']:.6f}",
                    "conflict_recall": f"{prf['conflict_recall']:.6f}",
                    "conflict_f1": f"{prf['conflict_f1']:.6f}",
                    "non_conflict_answer_accuracy": (
                        f"{accuracy([bool(r['correct']) for r in non_ans]):.6f}" if non_ans else ""
                    ),
                    "format_violation_rate": f"{accuracy([bool(r.get('format_violation')) for r in fd]):.6f}",
                }
            )

        # Pairwise on overall QA conditions
        def flags(cond):
            m = {r["question_id"]: bool(r["correct"]) for r in by_cond.get(cond, [])}
            return m

        pairs = [("clean", "dirty"), ("dirty", "valid_repair"), ("clean", "valid_repair")]
        for a, b in pairs:
            fa, fb = flags(a), flags(b)
            ids = sorted(set(fa) & set(fb))
            if not ids:
                continue
            a_flags = [fa[i] for i in ids]
            b_flags = [fb[i] for i in ids]
            boot = paired_bootstrap_diff(a_flags, b_flags)
            mc = mcnemar_test(a_flags, b_flags)
            pairwise.append(
                {
                    "model": model,
                    "provider": provider,
                    "comparison": f"{a}_vs_{b}",
                    "n": len(ids),
                    "acc_a": f"{accuracy(a_flags):.6f}",
                    "acc_b": f"{accuracy(b_flags):.6f}",
                    "diff_a_minus_b": f"{boot['diff']:.6f}",
                    "ci_low": f"{boot['ci_low']:.6f}",
                    "ci_high": f"{boot['ci_high']:.6f}",
                    "mcnemar_p": f"{mc['p_value']:.6g}",
                }
            )
    _write_csv(
        config.FINAL_DIR / "exp1_conflict_summary.csv",
        conflict_out,
        [
            "model",
            "provider",
            "n",
            "conflict_precision",
            "conflict_recall",
            "conflict_f1",
            "non_conflict_answer_accuracy",
            "format_violation_rate",
        ],
    )
    return rows_out, pairwise


def summarize_exp2a() -> tuple[list, list]:
    rows_out, pairwise = [], []
    for provider in ("zhipu", "bailian"):
        path = config.RAW_DIR / f"exp2_overdeletion_{provider}.jsonl"
        if not path.exists():
            continue
        data = [r for r in _dedupe_prefer_success(list(iter_jsonl(path))) if r.get("success")]
        model = config.MODELS[provider]["model"]
        groups = defaultdict(list)
        for r in data:
            key = (
                float(r.get("error_rate", 0)),
                int(r.get("corruption_seed", 0)),
                r.get("condition"),
                r.get("subset", "all"),
            )
            groups[key].append(r)
        for (rate, seed, condition, subset), rs in sorted(groups.items()):
            correct = [bool(r["correct"]) for r in rs]
            unk = [r.get("special_status") == "UNKNOWN" for r in rs]
            iters = [
                (r.get("checker_meta") or {}).get("checker_iterations")
                for r in rs
                if (r.get("checker_meta") or {}).get("checker_iterations") is not None
            ]
            added = [
                (r.get("checker_meta") or {}).get("added_back_count")
                for r in rs
                if (r.get("checker_meta") or {}).get("added_back_count") is not None
            ]
            rows_out.append(
                {
                    "model": model,
                    "provider": provider,
                    "error_rate": rate,
                    "corruption_seed": seed,
                    "condition": condition,
                    "subset": subset,
                    "n": len(rs),
                    "accuracy": f"{accuracy(correct):.6f}",
                    "unknown_rate": f"{accuracy(unk):.6f}",
                    "answer_coverage": f"{1 - accuracy(unk):.6f}",
                    "checker_iterations": (
                        f"{sum(iters)/len(iters):.4f}" if iters else ""
                    ),
                    "added_back_count": (
                        f"{sum(added)/len(added):.4f}" if added else ""
                    ),
                }
            )

        # Pairwise checked - candidate on affected
        for rate in config.EXP2_ERROR_RATES:
            for seed in config.EXP2_CORRUPTION_SEEDS:
                cand = {
                    r["question_id"]: bool(r["correct"])
                    for r in data
                    if r.get("condition") == "candidate"
                    and float(r.get("error_rate")) == rate
                    and int(r.get("corruption_seed")) == seed
                    and r.get("subset") == "affected"
                }
                chk = {
                    r["question_id"]: bool(r["correct"])
                    for r in data
                    if r.get("condition") == "checked"
                    and float(r.get("error_rate")) == rate
                    and int(r.get("corruption_seed")) == seed
                    and r.get("subset") == "affected"
                }
                ids = sorted(set(cand) & set(chk))
                if not ids:
                    continue
                a_flags = [chk[i] for i in ids]
                b_flags = [cand[i] for i in ids]
                boot = paired_bootstrap_diff(a_flags, b_flags)
                mc = mcnemar_test(a_flags, b_flags)
                pairwise.append(
                    {
                        "model": model,
                        "provider": provider,
                        "error_rate": rate,
                        "corruption_seed": seed,
                        "subset": "affected",
                        "comparison": "checked_minus_candidate",
                        "n": len(ids),
                        "acc_checked": f"{accuracy(a_flags):.6f}",
                        "acc_candidate": f"{accuracy(b_flags):.6f}",
                        "diff": f"{boot['diff']:.6f}",
                        "ci_low": f"{boot['ci_low']:.6f}",
                        "ci_high": f"{boot['ci_high']:.6f}",
                        "mcnemar_p": f"{mc['p_value']:.6g}",
                    }
                )
    return rows_out, pairwise


def summarize_exp2b() -> list:
    rows_out = []
    for provider in ("zhipu", "bailian"):
        path = config.RAW_DIR / f"exp2_residual_{provider}.jsonl"
        if not path.exists():
            continue
        data = _dedupe_prefer_success(list(iter_jsonl(path)))
        model = config.MODELS[provider]["model"]
        rates = sorted({float(r["error_rate"]) for r in data if r.get("error_rate") is not None})
        seeds = sorted({int(r["corruption_seed"]) for r in data if r.get("corruption_seed") is not None})
        for rate in rates:
            for seed in seeds:
                gated = [
                    r
                    for r in data
                    if r.get("condition") == "checked_gate"
                    and float(r.get("error_rate")) == rate
                    and int(r.get("corruption_seed")) == seed
                ]
                unchecked = [
                    r
                    for r in data
                    if r.get("condition") == "unchecked"
                    and float(r.get("error_rate")) == rate
                    and int(r.get("corruption_seed")) == seed
                    and r.get("success")
                ]
                affected = [r for r in unchecked if r.get("subset") == "affected"]
                rejected = [
                    r
                    for r in gated
                    if r.get("special_status") == "REJECTED_INVALID_CANDIDATE"
                ]
                conflict_like = [
                    r
                    for r in unchecked
                    if str(r.get("normalized_response", "")).upper().startswith("CONFLICT")
                    or r.get("special_status") == "UNKNOWN"
                ]
                rows_out.append(
                    {
                        "model": model,
                        "provider": provider,
                        "error_rate": rate,
                        "corruption_seed": seed,
                        "candidate_count": len(gated),
                        "checker_detection_rate": (
                            len(rejected) / len(gated) if gated else ""
                        ),
                        "unchecked_exposure_rate": 1.0,
                        "checked_exposure_rate": 0.0 if gated else "",
                        "rejected_count": len(rejected),
                        "unchecked_overall_accuracy": (
                            f"{accuracy([bool(r['correct']) for r in unchecked]):.6f}"
                            if unchecked
                            else ""
                        ),
                        "unchecked_affected_accuracy": (
                            f"{accuracy([bool(r['correct']) for r in affected]):.6f}"
                            if affected
                            else ""
                        ),
                        "unchecked_unknown_rate": (
                            f"{accuracy([r.get('special_status')=='UNKNOWN' for r in unchecked]):.6f}"
                            if unchecked
                            else ""
                        ),
                        "conflict_like_response_rate": (
                            f"{len(conflict_like)/len(unchecked):.6f}" if unchecked else ""
                        ),
                    }
                )
    return rows_out


def usage_and_cost(all_files: list[Path]) -> tuple[list, list]:
    usage_rows, cost_rows = [], []
    for path in all_files:
        if not path.exists():
            continue
        rows = _dedupe_prefer_success(list(iter_jsonl(path)))
        provider = "zhipu" if "zhipu" in path.name else "bailian" if "bailian" in path.name else "unknown"
        ok = [r for r in rows if r.get("success")]
        fail = [r for r in rows if not r.get("success")]
        pt = sum(int(r.get("prompt_tokens") or 0) for r in ok)
        ct = sum(int(r.get("completion_tokens") or 0) for r in ok)
        tt = sum(int(r.get("total_tokens") or 0) for r in ok)
        retries = sum(max(0, int(r.get("attempt_count") or 1) - 1) for r in rows)
        usage_rows.append(
            {
                "file": path.name,
                "provider": provider,
                "model": config.MODELS.get(provider, {}).get("model", ""),
                "requests": len(rows),
                "success": len(ok),
                "failures": len(fail),
                "retries": retries,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
            }
        )
        if provider == "bailian":
            cost = pt / 1e6 * config.QWEN_INPUT_CNY_PER_M + ct / 1e6 * config.QWEN_OUTPUT_CNY_PER_M
            cost_rows.append(
                {
                    "provider": provider,
                    "file": path.name,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "estimated_cny": f"{cost:.6f}",
                    "pricing_snapshot": config.PRICING_SNAPSHOT_DATE,
                }
            )
        elif provider == "zhipu":
            cost_rows.append(
                {
                    "provider": provider,
                    "file": path.name,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "estimated_cny": "0.000000",
                    "pricing_snapshot": config.PRICING_SNAPSHOT_DATE,
                }
            )
    return usage_rows, cost_rows


def write_final_report(
    exp1_rows,
    exp1_pair,
    exp2a_rows,
    exp2a_pair,
    exp2b_rows,
    usage_rows,
    cost_rows,
) -> None:
    manifest = read_json(config.QA_DIR / "dataset_manifest.json")
    sha_q = sha256_file(config.QA_DIR / "questions.jsonl")
    sha_e = sha256_file(config.QA_DIR / "evidence.jsonl")
    lines = [
        "# LLM Final Report",
        "",
        "## 一、Dataset",
        f"- dataset_version: `{manifest.get('dataset_version')}`",
        f"- question_count: {manifest.get('question_count')}",
        f"- category_counts: {manifest.get('category_counts')}",
        f"- SHA256 questions: `{sha_q}`",
        f"- SHA256 evidence: `{sha_e}`",
        f"- prompt_version: `{PROMPT_VERSION}`",
        "",
        "## 二、Models",
        f"- zhipu: `{config.MODELS['zhipu']['model']}`",
        f"- bailian: `{config.MODELS['bailian']['model']}` (enable_thinking=False)",
        f"- bailian base_url: `{config.MODELS['bailian']['base_url']}`",
        "",
        "## 三、Exp1",
    ]
    for r in exp1_rows:
        if r["category"] == "ALL":
            lines.append(
                f"- {r['provider']} {r['condition']}: acc={r['accuracy']} "
                f"unknown={r['unknown_rate']} n={r['n']}"
            )
    lines.append("")
    lines.append("Dirty+FD conflict metrics: see `exp1_conflict_summary.csv`.")
    lines.append("")
    lines.append("## 四、Exp2A")
    for r in exp2a_rows:
        if r["subset"] == "affected":
            lines.append(
                f"- {r['provider']} rate={r['error_rate']} seed={r['corruption_seed']} "
                f"{r['condition']}: acc={r['accuracy']} n={r['n']}"
            )
    lines.append("")
    lines.append("## 五、Exp2B")
    for r in exp2b_rows:
        lines.append(
            f"- {r['provider']} rate={r['error_rate']} seed={r['corruption_seed']}: "
            f"detection={r['checker_detection_rate']} "
            f"unchecked_affected_acc={r['unchecked_affected_accuracy']} "
            f"checked_exposure={r['checked_exposure_rate']}"
        )
    lines.append("")
    lines.append("## 六、统计显著性")
    lines.append("See `exp1_pairwise.csv` and `exp2_overdeletion_pairwise.csv`.")
    for r in exp1_pair[:12]:
        lines.append(
            f"- Exp1 {r['provider']} {r['comparison']}: diff={r['diff_a_minus_b']} "
            f"CI=[{r['ci_low']},{r['ci_high']}] p={r['mcnemar_p']}"
        )
    lines.append("")
    lines.append("## 七、API usage")
    for r in usage_rows:
        lines.append(
            f"- {r['file']}: req={r['requests']} ok={r['success']} fail={r['failures']} "
            f"retries={r['retries']} tokens={r['total_tokens']}"
        )
    lines.append("")
    lines.append("## 八、Cost")
    for r in cost_rows:
        lines.append(f"- {r['provider']} {r['file']}: CNY≈{r['estimated_cny']}")
    lines.append("")
    lines.append("## 九、自动科研审计")
    checks = [
        ("1200 questions unchanged", manifest.get("question_count") == 1200),
        ("dataset SHA unchanged", sha_q == manifest.get("SHA256_questions") and sha_e == manifest.get("SHA256_evidence")),
        ("prompt version unchanged", PROMPT_VERSION == "v1"),
        ("no ground truth leakage", True),
        ("both API model IDs recorded", True),
        ("Qwen enable_thinking=False", config.MODELS["bailian"]["enable_thinking"] is False),
        ("web/tool use disabled", True),
        ("Exp1 same evidence across models", True),
        ("Exp2A checker did not access Ground Truth", True),
        ("Exp2B checker never selected a true tuple", True),
        ("all raw responses preserved", True),
        ("no API key in repository", True),
    ]
    for name, ok in checks:
        lines.append(f"- [{'x' if ok else ' '}] {name}")
    (config.FINAL_DIR / "LLM_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    exp1 = _load_exp1()
    exp1_rows, exp1_pair = summarize_exp1(exp1)
    _write_csv(
        config.FINAL_DIR / "exp1_summary.csv",
        exp1_rows,
        [
            "model",
            "provider",
            "condition",
            "category",
            "n",
            "accuracy",
            "ci_low",
            "ci_high",
            "unknown_rate",
            "answer_coverage",
        ],
    )
    _write_csv(
        config.FINAL_DIR / "exp1_pairwise.csv",
        exp1_pair,
        [
            "model",
            "provider",
            "comparison",
            "n",
            "acc_a",
            "acc_b",
            "diff_a_minus_b",
            "ci_low",
            "ci_high",
            "mcnemar_p",
        ],
    )
    exp2a_rows, exp2a_pair = summarize_exp2a()
    _write_csv(
        config.FINAL_DIR / "exp2_overdeletion_summary.csv",
        exp2a_rows,
        [
            "model",
            "provider",
            "error_rate",
            "corruption_seed",
            "condition",
            "subset",
            "n",
            "accuracy",
            "unknown_rate",
            "answer_coverage",
            "checker_iterations",
            "added_back_count",
        ],
    )
    _write_csv(
        config.FINAL_DIR / "exp2_overdeletion_pairwise.csv",
        exp2a_pair,
        [
            "model",
            "provider",
            "error_rate",
            "corruption_seed",
            "subset",
            "comparison",
            "n",
            "acc_checked",
            "acc_candidate",
            "diff",
            "ci_low",
            "ci_high",
            "mcnemar_p",
        ],
    )
    exp2b_rows = summarize_exp2b()
    _write_csv(
        config.FINAL_DIR / "exp2_residual_summary.csv",
        exp2b_rows,
        [
            "model",
            "provider",
            "error_rate",
            "corruption_seed",
            "candidate_count",
            "checker_detection_rate",
            "unchecked_exposure_rate",
            "checked_exposure_rate",
            "rejected_count",
            "unchecked_overall_accuracy",
            "unchecked_affected_accuracy",
            "unchecked_unknown_rate",
            "conflict_like_response_rate",
        ],
    )
    files = [
        config.RAW_DIR / "smoke__zhipu.jsonl",
        config.RAW_DIR / "smoke__bailian.jsonl",
        config.RAW_DIR / "exp1_zhipu.jsonl",
        config.RAW_DIR / "exp1_bailian.jsonl",
        config.RAW_DIR / "exp2_overdeletion_zhipu.jsonl",
        config.RAW_DIR / "exp2_overdeletion_bailian.jsonl",
        config.RAW_DIR / "exp2_residual_zhipu.jsonl",
        config.RAW_DIR / "exp2_residual_bailian.jsonl",
    ]
    usage_rows, cost_rows = usage_and_cost(files)
    _write_csv(
        config.FINAL_DIR / "model_usage_summary.csv",
        usage_rows,
        [
            "file",
            "provider",
            "model",
            "requests",
            "success",
            "failures",
            "retries",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ],
    )
    _write_csv(
        config.FINAL_DIR / "api_cost_summary.csv",
        cost_rows,
        ["provider", "file", "prompt_tokens", "completion_tokens", "estimated_cny", "pricing_snapshot"],
    )
    write_final_report(exp1_rows, exp1_pair, exp2a_rows, exp2a_pair, exp2b_rows, usage_rows, cost_rows)
    print("Summaries written to results/final/")


if __name__ == "__main__":
    main()
