#!/usr/bin/env python
"""Validate the frozen dataset before any API spend."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from generation.database_generator import load_clean_database
from generation.dirty_generator import load_dirty_database
from generation.schemas import course_schema, enrollment_schema, student_schema
from utils.bootstrap import ensure_experiments_on_path
from utils.io_utils import iter_jsonl, read_json

ensure_experiments_on_path()
from common.fd_utils import find_fd_violation, satisfies_fds  # noqa: E402


def main() -> None:
    for name, fn in (
        ("STUDENT", student_schema),
        ("COURSE", course_schema),
        ("ENROLLMENT", enrollment_schema),
    ):
        fn()
        print(f"{name} BCNF: PASS")

    clean = load_clean_database(config.DATABASE_DIR / "clean")
    dirty = load_dirty_database(config.DATABASE_DIR / "dirty")

    for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
        schema = clean[rel]["schema"]
        assert satisfies_fds(clean[rel]["rows"], schema.fds, schema.attr_to_idx)
        assert not satisfies_fds(dirty[rel]["rows"], schema.fds, schema.attr_to_idx)
        assert find_fd_violation(dirty[rel]["rows"], schema.fds, schema.attr_to_idx)
    print("Clean FD consistency: PASS")

    verification = read_json(config.DATABASE_DIR / "valid_repair" / "verification.json")
    for rel, v in verification.items():
        assert v["candidate_consistent"] is True
        assert v["is_repair"] is True
    print("Valid Repair: PASS")

    questions = list(iter_jsonl(config.QA_DIR / "questions.jsonl"))
    evidence = {e["question_id"]: e for e in iter_jsonl(config.QA_DIR / "evidence.jsonl")}
    assert len(questions) == config.QUESTION_COUNT
    print(f"QA count: {len(questions)}")

    cats = {}
    for q in questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
        e = evidence[q["question_id"]]
        schema = clean[q["relation"]]["schema"]
        dirty_rows = [tuple(r[a] for a in schema.attributes) for r in e["dirty_evidence"]]
        if q["category"] == "No-Conflict":
            assert satisfies_fds(dirty_rows, schema.fds, schema.attr_to_idx)
        elif q["category"] == "Irrelevant-Conflict":
            assert find_fd_violation(dirty_rows, schema.fds, schema.attr_to_idx)
            # target key not conflicted
            key = q["query_key"]
            matched = [
                r for r in e["dirty_evidence"] if all(r.get(k) == v for k, v in key.items())
            ]
            assert len({tuple(m.items()) for m in matched}) == 1 or len(matched) == 1
        else:
            assert find_fd_violation(dirty_rows, schema.fds, schema.attr_to_idx)

    assert cats == config.CATEGORY_COUNTS
    print("Category distribution: PASS")

    over = read_json(config.CANDIDATE_DIR / "overdeletion.json")
    residual = read_json(config.CANDIDATE_DIR / "residual.json")
    for rate, rels in over["error_rates"].items():
        for rel, art in rels.items():
            assert art["initial_check"]["candidate_consistent"] is True
            assert art["initial_check"]["is_repair"] is False
            assert art["final_is_repair"] is True
    print("Over-deletion completion: PASS")

    for rate, rels in residual["error_rates"].items():
        for rel, art in rels.items():
            assert art["candidate_consistent"] is False
    print("Residual conflict detection: PASS")
    print("ALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
