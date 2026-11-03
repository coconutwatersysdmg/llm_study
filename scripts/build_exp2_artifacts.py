#!/usr/bin/env python
"""Build Exp2A/2B artifacts only (does NOT regenerate frozen QA dataset)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from generation.database_generator import load_clean_database
from generation.dirty_generator import load_dirty_database
from generation.candidate_repair_generator import build_valid_repair_of_dirty, save_valid_repair
from generation.exp2_protocol import (
    generate_overdeletion_protocol,
    generate_residual_protocol,
    save_json,
)
from utils.io_utils import iter_jsonl, read_json


def load_valid_repair(path: Path) -> dict:
    verification = read_json(path / "verification.json")
    relations = {}
    for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
        rows = []
        with (path / f"{rel.lower()}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        schema = read_json(path / f"{rel.lower()}_schema.json")
        relations[rel] = {"schema": schema, "rows": rows}
    return {"relations": relations, "verification": verification}


def main() -> None:
    dirty = load_dirty_database(config.DATABASE_DIR / "dirty")
    # Prefer existing valid_repair on disk (already verified); rebuild wrapper if needed
    vr_dir = config.DATABASE_DIR / "valid_repair"
    if (vr_dir / "verification.json").exists():
        valid = load_valid_repair(vr_dir)
    else:
        clean = load_clean_database(config.DATABASE_DIR / "clean")
        valid = build_valid_repair_of_dirty(clean, dirty)
        save_valid_repair(valid, vr_dir)

    questions = list(iter_jsonl(config.QA_DIR / "questions.jsonl"))
    over, eligible = generate_overdeletion_protocol(
        dirty,
        valid,
        questions,
        error_rates=config.EXP2_ERROR_RATES,
        corruption_seeds=config.EXP2_CORRUPTION_SEEDS,
    )
    residual = generate_residual_protocol(
        dirty,
        valid,
        questions,
        error_rates=config.EXP2_ERROR_RATES,
        corruption_seeds=config.EXP2_CORRUPTION_SEEDS,
    )

    save_json(config.CANDIDATE_DIR / "overdeletion_eligible_pool.json", eligible)
    save_json(config.CANDIDATE_DIR / "overdeletion.json", over)
    save_json(config.CANDIDATE_DIR / "exp2a_question_manifest.json", over["question_manifest"])
    save_json(config.CANDIDATE_DIR / "residual.json", residual)
    save_json(config.CANDIDATE_DIR / "exp2b_question_manifest.json", residual["question_manifest"])

    print(f"eligible_pool_total={eligible['total']}")
    print(f"eligible_counts={ {k: v['count'] for k,v in eligible['relations'].items()} }")
    print(f"exp2a_runs={len(over['question_manifest']['runs'])}")
    print(f"exp2b_runs={len(residual['question_manifest']['runs'])}")
    print("Exp2 artifacts built (QA dataset untouched).")


if __name__ == "__main__":
    main()
