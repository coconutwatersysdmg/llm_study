"""QA / evidence category tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation.database_generator import generate_clean_database
from generation.dirty_generator import generate_dirty_database
from generation.qa_generator import generate_qa_dataset
from prompts.templates import PromptBuilder
from utils.bootstrap import ensure_experiments_on_path

ensure_experiments_on_path()
from common.fd_utils import find_fd_violation, satisfies_fds  # noqa: E402


def _mini_qa():
    clean = generate_clean_database(n_students=60, n_courses=15, n_enrollments=300, seed=7)
    dirty = generate_dirty_database(
        clean,
        seed=7,
        conflicts_per_relation={"STUDENT": 20, "COURSE": 8, "ENROLLMENT": 30},
    )
    cats = {
        "No-Conflict": 12,
        "Irrelevant-Conflict": 12,
        "Answer-Critical-Conflict": 16,
    }
    questions, evidence = generate_qa_dataset(clean, dirty, cats, seed=7, evidence_order_seed=7)
    ev = {e["question_id"]: e for e in evidence}
    return clean, questions, ev


def test_qa_categories_and_evidence() -> None:
    clean, questions, ev = _mini_qa()
    assert len(questions) == 40
    for q in questions:
        e = ev[q["question_id"]]
        schema = clean[q["relation"]]["schema"]
        dirty_rows = [tuple(r[a] for a in schema.attributes) for r in e["dirty_evidence"]]
        clean_rows = [tuple(r[a] for a in schema.attributes) for r in e["clean_evidence"]]
        assert satisfies_fds(clean_rows, schema.fds, schema.attr_to_idx)
        key = q["query_key"]
        if q["category"] == "No-Conflict":
            assert satisfies_fds(dirty_rows, schema.fds, schema.attr_to_idx)
            assert find_fd_violation(dirty_rows, schema.fds, schema.attr_to_idx) is None
        elif q["category"] == "Irrelevant-Conflict":
            assert find_fd_violation(dirty_rows, schema.fds, schema.attr_to_idx) is not None
            matched = [
                r for r in e["dirty_evidence"] if all(r.get(k) == v for k, v in key.items())
            ]
            # target itself should be unique among matched contents
            assert len({tuple(sorted(m.items())) for m in matched}) == 1
        else:
            assert find_fd_violation(dirty_rows, schema.fds, schema.attr_to_idx) is not None
            matched = [
                r for r in e["dirty_evidence"] if all(r.get(k) == v for k, v in key.items())
            ]
            assert len({tuple(sorted(m.items())) for m in matched}) >= 2


def test_no_ground_truth_leakage() -> None:
    _, questions, ev = _mini_qa()
    builder = PromptBuilder()
    for q in questions[:10]:
        e = ev[q["question_id"]]
        sys_p, user_p = builder.qa_prompts(q["question"], e["dirty_evidence"])
        builder.assert_no_leakage(sys_p, user_p)
        # Answer value might coincidentally appear in evidence; field names must not.
        assert '"answer"' not in user_p
        assert "clean_source_label" not in user_p
        assert "expected_status" not in user_p
        sys_p2, user_p2 = builder.conflict_prompts(q["relation"], q["question"], e["dirty_evidence"])
        builder.assert_no_leakage(sys_p2, user_p2)
