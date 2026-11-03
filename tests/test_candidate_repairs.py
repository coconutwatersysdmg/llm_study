"""Candidate repair artifact tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation.candidate_repair_generator import (
    build_valid_repair_of_dirty,
    generate_overdeletion_candidates,
    generate_residual_candidates,
)
from generation.database_generator import generate_clean_database
from generation.dirty_generator import generate_dirty_database
from generation.qa_generator import generate_qa_dataset


def test_valid_repair_flags() -> None:
    clean = generate_clean_database(n_students=40, n_courses=10, n_enrollments=150, seed=3)
    dirty = generate_dirty_database(
        clean, seed=3, conflicts_per_relation={"STUDENT": 8, "COURSE": 4, "ENROLLMENT": 12}
    )
    valid = build_valid_repair_of_dirty(clean, dirty)
    for rel, v in valid["verification"].items():
        assert v["candidate_consistent"] is True
        assert v["is_repair"] is True


def test_overdeletion_and_residual() -> None:
    clean = generate_clean_database(n_students=40, n_courses=10, n_enrollments=150, seed=5)
    dirty = generate_dirty_database(
        clean, seed=5, conflicts_per_relation={"STUDENT": 8, "COURSE": 4, "ENROLLMENT": 12}
    )
    valid = build_valid_repair_of_dirty(clean, dirty)
    questions, _ = generate_qa_dataset(
        clean,
        dirty,
        {"No-Conflict": 6, "Irrelevant-Conflict": 6, "Answer-Critical-Conflict": 8},
        seed=5,
    )
    over = generate_overdeletion_candidates(
        dirty, valid, questions, error_rates=[0.05, 0.10], seed=5
    )
    for rate, rels in over["error_rates"].items():
        for rel, art in rels.items():
            assert art["initial_check"]["candidate_consistent"] is True
            assert art["initial_check"]["is_repair"] is False
            assert art["final_is_repair"] is True
            assert art["final_candidate_consistent"] is True

    residual = generate_residual_candidates(dirty, valid, error_rates=[0.05], seed=5)
    for rate, rels in residual["error_rates"].items():
        for rel, art in rels.items():
            assert art["candidate_consistent"] is False
            assert art["is_repair"] is False
