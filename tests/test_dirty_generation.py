"""Dirty generation tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation.database_generator import generate_clean_database
from generation.dirty_generator import generate_dirty_database
from utils.bootstrap import ensure_experiments_on_path

ensure_experiments_on_path()
from common.fd_utils import find_fd_violation, satisfies_fds  # noqa: E402


def test_dirty_contains_expected_conflicts() -> None:
    clean = generate_clean_database(n_students=80, n_courses=20, n_enrollments=400, seed=42)
    dirty = generate_dirty_database(
        clean,
        seed=42,
        conflicts_per_relation={"STUDENT": 10, "COURSE": 5, "ENROLLMENT": 15},
    )
    assert len(dirty["conflicts"]) == 30
    for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
        schema = dirty[rel]["schema"]
        assert not satisfies_fds(dirty[rel]["rows"], schema.fds, schema.attr_to_idx)
        assert find_fd_violation(dirty[rel]["rows"], schema.fds, schema.attr_to_idx) is not None
    for c in dirty["conflicts"]:
        assert c["clean_row"] != c["conflicting_row"]
        assert c["changed_attributes"]
        for a, v in c["key"].items():
            assert c["clean_row"][a] == v
            assert c["conflicting_row"][a] == v
