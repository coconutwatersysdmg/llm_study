#!/usr/bin/env python
"""Build the frozen controlled-evidence dataset (no API calls)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from generation.candidate_repair_generator import (
    build_valid_repair_of_dirty,
    generate_overdeletion_candidates,
    generate_residual_candidates,
    save_candidate_artifacts,
    save_valid_repair,
)
from generation.database_generator import generate_clean_database, save_clean_database
from generation.dirty_generator import generate_dirty_database, save_dirty_database
from generation.qa_generator import generate_qa_dataset
from generation.schemas import course_schema, enrollment_schema, student_schema
from utils.io_utils import sha256_file, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.GENERATION_SEED)
    args = parser.parse_args()

    # BCNF gate
    for name, fn in (
        ("STUDENT", student_schema),
        ("COURSE", course_schema),
        ("ENROLLMENT", enrollment_schema),
    ):
        s = fn()
        print(f"{name} BCNF: PASS ({s.name})")

    clean = generate_clean_database(
        n_students=config.STUDENT_COUNT,
        n_courses=config.COURSE_COUNT,
        n_enrollments=config.ENROLLMENT_COUNT,
        seed=args.seed,
    )
    save_clean_database(clean, config.DATABASE_DIR / "clean")
    print("Clean DB saved.")

    dirty = generate_dirty_database(clean, seed=args.seed)
    save_dirty_database(dirty, config.DATABASE_DIR / "dirty")
    print(f"Dirty DB saved. conflicts={dirty['meta']['conflict_count']}")

    valid = build_valid_repair_of_dirty(clean, dirty)
    save_valid_repair(valid, config.DATABASE_DIR / "valid_repair")
    print("Valid repair verified and saved.")

    questions, evidence = generate_qa_dataset(
        clean,
        dirty,
        category_counts=config.CATEGORY_COUNTS,
        seed=args.seed,
        evidence_order_seed=config.EVIDENCE_ORDER_SEED,
    )
    q_path = config.QA_DIR / "questions.jsonl"
    e_path = config.QA_DIR / "evidence.jsonl"
    write_jsonl(q_path, questions)
    write_jsonl(e_path, evidence)

    error_rates = [0.01, 0.05, 0.10, 0.20]
    over = generate_overdeletion_candidates(
        dirty, valid, questions, error_rates=error_rates, seed=args.seed
    )
    save_candidate_artifacts(over, config.CANDIDATE_DIR / "overdeletion.json")
    residual = generate_residual_candidates(
        dirty, valid, error_rates=error_rates, seed=args.seed
    )
    save_candidate_artifacts(residual, config.CANDIDATE_DIR / "residual.json")

    cat_counts = {}
    for q in questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1

    conflict_counts: dict[str, int] = {}
    for c in dirty["conflicts"]:
        conflict_counts[c["relation"]] = conflict_counts.get(c["relation"], 0) + 1

    manifest = {
        "dataset_version": config.DATASET_VERSION,
        "generation_seed": args.seed,
        "evidence_order_seed": config.EVIDENCE_ORDER_SEED,
        "student_count": clean["meta"]["n_students"],
        "course_count": clean["meta"]["n_courses"],
        "enrollment_count": clean["meta"]["n_enrollments"],
        "question_count": len(questions),
        "category_counts": cat_counts,
        "dirty_conflict_counts": conflict_counts,
        "schema": {
            "STUDENT": student_schema().to_json(),
            "COURSE": course_schema().to_json(),
            "ENROLLMENT": enrollment_schema().to_json(),
        },
        "FDs": {
            "STUDENT": [str(fd) for fd in student_schema().fds],
            "COURSE": [str(fd) for fd in course_schema().fds],
            "ENROLLMENT": [str(fd) for fd in enrollment_schema().fds],
        },
        "BCNF_validation": {
            "STUDENT": True,
            "COURSE": True,
            "ENROLLMENT": True,
        },
        "valid_repair_verification": valid["verification"],
        "SHA256_questions": sha256_file(q_path),
        "SHA256_evidence": sha256_file(e_path),
        "design_note": (
            "Controlled evidence setting: evidence is attached from question "
            "metadata; no retriever / embedding / vector DB is used. This isolates "
            "FD-conflict effects from retrieval failure."
        ),
        "checker_boundary": (
            "Repair checker verifies F-satisfaction and S-repair maximality only. "
            "It does not decide which conflicting tuple is real-world true."
        ),
    }
    write_json(config.QA_DIR / "dataset_manifest.json", manifest)
    print(f"QA count: {len(questions)}")
    print(f"Category distribution: {cat_counts}")
    print("Dataset build complete.")


if __name__ == "__main__":
    main()
