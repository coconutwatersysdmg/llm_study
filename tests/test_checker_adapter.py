"""Checker adapter tests — including no-GT completion boundary."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checker.repair_checker_adapter import (
    check_candidate,
    complete_overdeleted_candidate,
    verify_valid_repair,
)
from generation.database_generator import generate_clean_database
from generation.dirty_generator import generate_dirty_database, make_conflicting_row
from utils.bootstrap import ensure_experiments_on_path

ensure_experiments_on_path()


def test_valid_repair_via_adapter() -> None:
    clean = generate_clean_database(n_students=30, n_courses=8, n_enrollments=100, seed=1)
    dirty = generate_dirty_database(
        clean, seed=1, conflicts_per_relation={"STUDENT": 5, "COURSE": 3, "ENROLLMENT": 8}
    )
    schema = clean["STUDENT"]["schema"]
    res = verify_valid_repair(schema, dirty["STUDENT"]["rows"], clean["STUDENT"]["rows"])
    assert res.candidate_consistent is True
    assert res.is_repair is True


def test_overdeleted_completion_without_gt_in_source() -> None:
    src = inspect.getsource(complete_overdeleted_candidate)
    forbidden = ("ground_truth", "clean_answer", "clean_source", "expected_status", "answer")
    for tok in forbidden:
        # allow the docstring mention of "ground truth" / "answers"
        pass
    # Stronger: function body after docstring should not reference question/answer fields
    body = src.split('"""', 2)[-1]
    assert "ground_truth" not in body
    assert "clean_source_label" not in body
    assert "expected_status" not in body
    assert "question" not in body

    clean = generate_clean_database(n_students=30, n_courses=8, n_enrollments=100, seed=2)
    dirty = generate_dirty_database(
        clean, seed=2, conflicts_per_relation={"STUDENT": 5, "COURSE": 3, "ENROLLMENT": 8}
    )
    schema = clean["STUDENT"]["schema"]
    r = dirty["STUDENT"]["rows"]
    r_prime = list(clean["STUDENT"]["rows"])
    removed = r_prime.pop()
    pre = check_candidate(schema, r, r_prime)
    assert pre.candidate_consistent is True
    assert pre.is_repair is False
    out = complete_overdeleted_candidate(schema, r, r_prime)
    assert out["final_is_repair"] is True
    assert removed in out["r_prime"] or out["added_back_count"] >= 1


def test_residual_detected_not_auto_fixed() -> None:
    clean = generate_clean_database(n_students=20, n_courses=5, n_enrollments=50, seed=9)
    schema = clean["STUDENT"]["schema"]
    r_prime = list(clean["STUDENT"]["rows"])
    twin, _ = make_conflicting_row(schema, r_prime[0], __import__("random").Random(0), "T")
    candidate = r_prime + [twin]
    universe = candidate
    res = check_candidate(schema, universe, candidate)
    assert res.candidate_consistent is False
    try:
        complete_overdeleted_candidate(schema, universe, candidate)
        assert False, "should raise on residual conflict"
    except ValueError:
        pass
