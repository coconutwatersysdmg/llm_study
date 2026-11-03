#!/usr/bin/env python
"""Estimate API cost before spending."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from evaluation.runner import estimate_tokens
from generation.evidence_builder import format_evidence_block
from prompts.templates import PromptBuilder
from utils.io_utils import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["exp1", "exp2a", "exp2b", "smoke", "all"],
        default="all",
    )
    parser.add_argument("--questions", type=int, default=None)
    args = parser.parse_args()

    questions = list(iter_jsonl(config.QA_DIR / "questions.jsonl"))
    evidence = {e["question_id"]: e for e in iter_jsonl(config.QA_DIR / "evidence.jsonl")}
    if args.questions:
        questions = questions[: args.questions]

    builder = PromptBuilder()
    n = len(questions)

    def estimate_condition(condition: str, conflict: bool = False) -> int:
        total = 0
        for q in questions:
            e = evidence[q["question_id"]]
            if condition == "clean":
                ev = e["clean_evidence"]
            elif condition == "dirty":
                ev = e["dirty_evidence"]
            else:
                ev = e["valid_repair_evidence"]
            if conflict:
                sys_p, user_p = builder.conflict_prompts(q["relation"], q["question"], e["dirty_evidence"])
            else:
                sys_p, user_p = builder.qa_prompts(q["question"], ev)
            total += estimate_tokens(sys_p + user_p)
        return total

    plans = {}
    if args.experiment in ("exp1", "all"):
        # 2 models × 4 conditions × n
        inp = (
            estimate_condition("clean")
            + estimate_condition("dirty")
            + estimate_condition("valid_repair")
            + estimate_condition("dirty", conflict=True)
        )
        plans["exp1"] = {
            "planned_request_count": 2 * 4 * n,
            "estimated_input_tokens_per_model": inp,
            "estimated_input_tokens_total": 2 * inp,
            "maximum_output_tokens": 2 * 4 * n * config.MAX_TOKENS,
        }
    if args.experiment in ("smoke", "all"):
        n_smoke = min(100, n)
        # scale roughly
        scale = n_smoke / max(n, 1)
        plans["smoke"] = {
            "planned_request_count": 1 * 4 * n_smoke,
            "note": "1 model (zhipu) × 4 conditions × 100 questions",
            "approx_input_tokens": int(
                (
                    estimate_condition("clean")
                    + estimate_condition("dirty")
                    + estimate_condition("valid_repair")
                    + estimate_condition("dirty", conflict=True)
                )
                * scale
            ),
            "maximum_output_tokens": 4 * n_smoke * config.MAX_TOKENS,
        }
    if args.experiment in ("exp2a", "all"):
        # 2 models × 4 rates × 2 conditions × n
        plans["exp2a"] = {
            "planned_request_count": 2 * 4 * 2 * n,
            "note": "OverDeleted vs CheckerCompleted; evidence size ~ valid_repair",
            "approx_input_tokens": 2 * 4 * 2 * estimate_condition("valid_repair"),
            "maximum_output_tokens": 2 * 4 * 2 * n * config.MAX_TOKENS,
        }
    if args.experiment in ("exp2b", "all"):
        # Unchecked residual calls LLM; Checked Gate does not if rejected.
        plans["exp2b"] = {
            "planned_request_count_llm": 2 * 4 * n,  # unchecked only
            "planned_checker_only": 2 * 4 * n,
            "note": "Checked Gate rejects inconsistent candidates without LLM calls.",
            "approx_input_tokens": 2 * 4 * estimate_condition("dirty"),
            "maximum_output_tokens": 2 * 4 * n * config.MAX_TOKENS,
        }

    # Cost for Qwen only (GLM treated free)
    qwen_in = config.QWEN_INPUT_CNY_PER_M
    qwen_out = config.QWEN_OUTPUT_CNY_PER_M
    print("pricing_snapshot_date:", config.PRICING_SNAPSHOT_DATE)
    print(f"QWEN_INPUT_CNY_PER_M={qwen_in} QWEN_OUTPUT_CNY_PER_M={qwen_out}")
    print(f"GLM-4-Flash-250414 treated as free API.")
    for name, plan in plans.items():
        print(f"\n=== {name} ===")
        for k, v in plan.items():
            print(f"  {k}: {v}")
        if "estimated_input_tokens_total" in plan:
            half = plan["estimated_input_tokens_total"] / 2
            out = plan["maximum_output_tokens"] / 2
            cost = half / 1e6 * qwen_in + out / 1e6 * qwen_out
            print(f"  estimated_qwen_cost_cny_upper: {cost:.4f}")
        elif "approx_input_tokens" in plan and name != "smoke":
            # half of total if 2 models
            half = plan["approx_input_tokens"] / 2
            out = plan["maximum_output_tokens"] / 2
            cost = half / 1e6 * qwen_in + out / 1e6 * qwen_out
            print(f"  estimated_qwen_cost_cny_upper: {cost:.4f}")


if __name__ == "__main__":
    main()
