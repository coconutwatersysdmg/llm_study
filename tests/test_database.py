"""Tests for BCNF schemas and clean database."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation.database_generator import generate_clean_database
from generation.schemas import course_schema, enrollment_schema, student_schema
from utils.bootstrap import ensure_experiments_on_path

ensure_experiments_on_path()
from common.fd_utils import attribute_closure, satisfies_fds, validate_bcnf  # noqa: E402


def test_three_relations_bcnf() -> None:
    for fn in (student_schema, course_schema, enrollment_schema):
        s = fn()
        assert validate_bcnf(s.U, s.fds)
        s.validate_bcnf()


def test_attribute_closure_student() -> None:
    s = student_schema()
    assert attribute_closure({"student_id"}, s.fds) == s.U


def test_clean_satisfies_fds() -> None:
    db = generate_clean_database(n_students=50, n_courses=10, n_enrollments=200, seed=42)
    for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
        schema = db[rel]["schema"]
        assert satisfies_fds(db[rel]["rows"], schema.fds, schema.attr_to_idx)
