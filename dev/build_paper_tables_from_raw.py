#!/usr/bin/env python
"""Build paper tables from final LLM summaries (no new API calls)."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

LLM_ROOT = Path(__file__).resolve().parents[1]
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

import config
from utils.io_utils import ensure_dir, iter_jsonl, read_json

# Tables live under experiments/ next to paper_figures
EXPERIMENTS_ROOT = config.PROJECT_ROOT.parent / "experiments"
TABLE_DIR = EXPERIMENTS_ROOT / "paper_tables"


def _dedupe(rows):
    best = {}
    order = []
    for r in rows:
        rid = r["request_id"]
        if rid not in best:
            best[rid] = r
            order.append(rid)
        elif (not best[rid].get("success")) and r.get("success"):
            best[rid] = r
    return [best[i] for i in order]


def accuracy(flags):
    return sum(1 for x in flags if x) / len(flags) if flags else float("nan")


def build_checker_table() -> list[dict]:
    rows_out = []
    # Part A: over-deletion aggregate across rates×seeds
    for provider, model in (
        ("zhipu", "GLM-4-Flash"),
        ("bailian", "Qwen3.7-Flash"),
    ):
        path = config.RAW_DIR / f"exp2_overdeletion_{provider}.jsonl"
        data = [r for r in _dedupe(list(iter_jsonl(path))) if r.get("success")]
        buckets = defaultdict(list)
        for r in data:
            buckets[(r.get("condition"), r.get("subset"))].append(r)
        def met(cond, subset, field="correct"):
            rs = buckets.get((cond, subset), [])
            if field == "correct":
                return accuracy([bool(r["correct"]) for r in rs])
            if field == "unknown":
                return accuracy([r.get("special_status") == "UNKNOWN" for r in rs])
            return float("nan")

        # Part B residual weighted
        rpath = config.RAW_DIR / f"exp2_residual_{provider}.jsonl"
        rdata = _dedupe(list(iter_jsonl(rpath)))
        gated = [r for r in rdata if r.get("condition") == "checked_gate"]
        unchecked = [r for r in rdata if r.get("condition") == "unchecked" and r.get("success")]
        affected = [r for r in unchecked if r.get("subset") == "affected"]
        rejected = [r for r in gated if r.get("special_status") == "REJECTED_INVALID_CANDIDATE"]

        # weight by candidate_count per rate×seed group
        groups = defaultdict(list)
        for r in affected:
            groups[(float(r["error_rate"]), int(r["corruption_seed"]))].append(r)
        w_num = w_den = 0.0
        for key, rs in groups.items():
            w = len(rs)
            w_num += accuracy([bool(r["correct"]) for r in rs]) * w
            w_den += w
        weighted_aff = w_num / w_den if w_den else float("nan")

        rows_out.append(
            {
                "part": "A_overdeletion",
                "model": model,
                "provider": provider,
                "candidate_affected_accuracy": f"{met('candidate','affected'):.6f}",
                "checked_affected_accuracy": f"{met('checked','affected'):.6f}",
                "candidate_unaffected_accuracy": f"{met('candidate','unaffected'):.6f}",
                "checked_unaffected_accuracy": f"{met('checked','unaffected'):.6f}",
                "candidate_affected_unknown_rate": f"{met('candidate','affected','unknown'):.6f}",
                "checked_affected_unknown_rate": f"{met('checked','affected','unknown'):.6f}",
                "checker_detection_rate": "",
                "unchecked_exposure_rate": "",
                "checked_exposure_rate": "",
                "unchecked_affected_qa_accuracy": "",
            }
        )
        rows_out.append(
            {
                "part": "B_residual",
                "model": model,
                "provider": provider,
                "candidate_affected_accuracy": "",
                "checked_affected_accuracy": "",
                "candidate_unaffected_accuracy": "",
                "checked_unaffected_accuracy": "",
                "candidate_affected_unknown_rate": "",
                "checked_affected_unknown_rate": "",
                "checker_detection_rate": f"{(len(rejected)/len(gated) if gated else float('nan')):.6f}",
                "unchecked_exposure_rate": "1.000000",
                "checked_exposure_rate": "0.000000",
                "unchecked_affected_qa_accuracy": f"{weighted_aff:.6f}",
            }
        )
    return rows_out


def build_fd_table() -> list[dict]:
    path = config.FINAL_DIR / "exp1_conflict_summary.csv"
    out = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(
                {
                    "Model": "GLM-4-Flash" if r["provider"] == "zhipu" else "Qwen3.7-Flash",
                    "Conflict Precision": r["conflict_precision"],
                    "Conflict Recall": r["conflict_recall"],
                    "Conflict F1": r["conflict_f1"],
                    "Non-conflict Answer Accuracy": r["non_conflict_answer_accuracy"],
                }
            )
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_stats_md(path: Path) -> None:
    pair = list(csv.DictReader((config.FINAL_DIR / "exp1_pairwise.csv").open(encoding="utf-8")))
    summ = list(csv.DictReader((config.FINAL_DIR / "exp1_summary.csv").open(encoding="utf-8")))
    conf = list(csv.DictReader((config.FINAL_DIR / "exp1_conflict_summary.csv").open(encoding="utf-8")))
    usage = list(csv.DictReader((config.FINAL_DIR / "model_usage_summary.csv").open(encoding="utf-8")))
    checker = list(csv.DictReader((TABLE_DIR / "table_llm_checker.csv").open(encoding="utf-8")))

    def find_sum(prov, cond, cat):
        for r in summ:
            if r["provider"] == prov and r["condition"] == cond and r["category"] == cat:
                return r
        return {}

    lines = [
        "# LLM Paper Statistics (auto-generated; no interpretation)",
        "",
        "## 1–4. Exp1 paired comparisons",
    ]
    for r in pair:
        lines.append(
            f"- {r['provider']} {r['comparison']}: "
            f"acc_a={r['acc_a']} acc_b={r['acc_b']} "
            f"diff={r['diff_a_minus_b']} CI=[{r['ci_low']},{r['ci_high']}] "
            f"McNemar_p={r['mcnemar_p']}"
        )
    lines.append("")
    lines.append("## 5. Answer-Critical Dirty accuracy")
    for prov in ("zhipu", "bailian"):
        r = find_sum(prov, "dirty", "Answer-Critical-Conflict")
        lines.append(f"- {prov}: accuracy={r.get('accuracy')} n={r.get('n')}")
    lines.append("")
    lines.append("## 6. Dirty+FD P/R/F1")
    for r in conf:
        lines.append(
            f"- {r['provider']}: P={r['conflict_precision']} R={r['conflict_recall']} "
            f"F1={r['conflict_f1']} non_conflict_acc={r['non_conflict_answer_accuracy']}"
        )
    lines.append("")
    lines.append("## 7. Over-deletion affected Candidate vs Checked")
    for r in checker:
        if r["part"] == "A_overdeletion":
            lines.append(
                f"- {r['model']}: Candidate-Affected={r['candidate_affected_accuracy']} "
                f"Checked-Affected={r['checked_affected_accuracy']} "
                f"Cand-Unaffected={r['candidate_unaffected_accuracy']} "
                f"Chk-Unaffected={r['checked_unaffected_accuracy']}"
            )
    lines.append("")
    lines.append("## 8. Residual checker detection / exposure")
    for r in checker:
        if r["part"] == "B_residual":
            lines.append(
                f"- {r['model']}: detection={r['checker_detection_rate']} "
                f"unchecked_exposure={r['unchecked_exposure_rate']} "
                f"checked_exposure={r['checked_exposure_rate']} "
                f"unchecked_affected_acc={r['unchecked_affected_qa_accuracy']}"
            )
    lines.append("")
    lines.append("## 9. API completeness (formal files)")
    for r in usage:
        if r["file"].startswith(("exp1_", "exp2_")):
            lines.append(
                f"- {r['file']}: success={r['success']}/{r['requests']} failures={r['failures']}"
            )
    # Exp1 n check
    lines.append("")
    lines.append("## Exp1 n completeness (ALL)")
    for prov in ("zhipu", "bailian"):
        for cond in ("clean", "dirty", "valid_repair"):
            r = find_sum(prov, cond, "ALL")
            lines.append(f"- {prov} {cond}: n={r.get('n')} acc={r.get('accuracy')}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(TABLE_DIR)
    checker_rows = build_checker_table()
    write_csv(
        TABLE_DIR / "table_llm_checker.csv",
        checker_rows,
        [
            "part",
            "model",
            "provider",
            "candidate_affected_accuracy",
            "checked_affected_accuracy",
            "candidate_unaffected_accuracy",
            "checked_unaffected_accuracy",
            "candidate_affected_unknown_rate",
            "checked_affected_unknown_rate",
            "checker_detection_rate",
            "unchecked_exposure_rate",
            "checked_exposure_rate",
            "unchecked_affected_qa_accuracy",
        ],
    )
    fd_rows = build_fd_table()
    write_csv(
        TABLE_DIR / "table_fd_prompt.csv",
        fd_rows,
        [
            "Model",
            "Conflict Precision",
            "Conflict Recall",
            "Conflict F1",
            "Non-conflict Answer Accuracy",
        ],
    )
    write_stats_md(TABLE_DIR / "llm_paper_statistics.md")
    print(f"Wrote tables under {TABLE_DIR}")


if __name__ == "__main__":
    main()
