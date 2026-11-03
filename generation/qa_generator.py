"""QA dataset generator with controlled evidence categories."""

from __future__ import annotations

import random
from typing import Any

from generation.database_generator import row_to_dict
from generation.dirty_generator import key_dict, make_conflicting_row
from generation.evidence_builder import (
    same_field_order,
    sample_distractors,
    shuffle_pair,
)
from utils.bootstrap import ensure_experiments_on_path

ensure_experiments_on_path()

from common.fd_utils import find_fd_violation, project_row, satisfies_fds  # noqa: E402
from common.types import RelationSchema, Row  # noqa: E402

QUESTION_TEMPLATES = [
    {
        "relation": "STUDENT",
        "template": "学生 {student_id} 的专业是什么？",
        "query_key": ("student_id",),
        "answer_attr": "major",
    },
    {
        "relation": "STUDENT",
        "template": "学生 {student_id} 的年级是什么？",
        "query_key": ("student_id",),
        "answer_attr": "grade_level",
    },
    {
        "relation": "COURSE",
        "template": "课程 {course_id} 的学分是多少？",
        "query_key": ("course_id",),
        "answer_attr": "credits",
    },
    {
        "relation": "COURSE",
        "template": "课程 {course_id} 属于哪个院系？",
        "query_key": ("course_id",),
        "answer_attr": "department",
    },
    {
        "relation": "ENROLLMENT",
        "template": "学生 {student_id} 在课程 {course_id} 中的成绩是多少？",
        "query_key": ("student_id", "course_id"),
        "answer_attr": "score",
    },
]


def _rows_as_dicts(schema: RelationSchema, rows: list[Row]) -> list[dict[str, Any]]:
    return [row_to_dict(schema, r) for r in rows]


def _lookup_row(
    schema: RelationSchema,
    rows: list[Row],
    key_attrs: tuple[str, ...],
    key_vals: dict[str, Any],
) -> Row | None:
    for row in rows:
        if all(row[schema.attr_to_idx[a]] == key_vals[a] for a in key_attrs):
            return row
    return None


def _evidence_has_violation(schema: RelationSchema, evidence: list[dict[str, Any]]) -> bool:
    rows = [tuple(rec[a] for a in schema.attributes) for rec in evidence]
    return find_fd_violation(rows, schema.fds, schema.attr_to_idx) is not None


def _target_key_conflicted(
    schema: RelationSchema,
    evidence: list[dict[str, Any]],
    key_vals: dict[str, Any],
) -> bool:
    key_attrs = schema.candidate_keys[0]
    matched = [
        rec
        for rec in evidence
        if all(rec[a] == key_vals[a] for a in key_attrs)
    ]
    if len(matched) < 2:
        return False
    # Same key, different non-key values ⇒ conflict on target
    first = matched[0]
    return any(rec != first for rec in matched[1:])


def generate_qa_dataset(
    clean_db: dict[str, Any],
    dirty_db: dict[str, Any],
    category_counts: dict[str, int],
    seed: int = 42,
    evidence_order_seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (questions, evidence_records) with fixed IDs Q000001…"""
    rng = random.Random(seed)
    conflict_by_rel: dict[str, list[dict[str, Any]]] = {
        "STUDENT": [],
        "COURSE": [],
        "ENROLLMENT": [],
    }
    for c in dirty_db["conflicts"]:
        conflict_by_rel[c["relation"]].append(c)

    questions: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    q_index = 0

    plan: list[str] = []
    for cat, n in category_counts.items():
        plan.extend([cat] * n)
    rng.shuffle(plan)

    # Round-robin templates with deterministic entity selection.
    for category in plan:
        q_index += 1
        qid = f"Q{q_index:06d}"
        tmpl = QUESTION_TEMPLATES[(q_index - 1) % len(QUESTION_TEMPLATES)]
        rel = tmpl["relation"]
        schema: RelationSchema = clean_db[rel]["schema"]
        clean_rows: list[Row] = clean_db[rel]["rows"]
        key_attrs = tmpl["query_key"]
        answer_attr = tmpl["answer_attr"]
        field_order = schema.attributes

        if category == "Answer-Critical-Conflict":
            confs = conflict_by_rel[rel]
            if not confs:
                raise RuntimeError(f"No conflicts available for {rel}")
            conf = confs[(q_index - 1) % len(confs)]
            key_vals = dict(conf["key"])
            clean_row = tuple(conf["clean_row"][a] for a in field_order)
            dirty_row = tuple(conf["conflicting_row"][a] for a in field_order)
        else:
            # Prefer keys that are NOT in conflict set for No/Irrelevant.
            conflict_keys = {
                tuple(c["key"][a] for a in schema.candidate_keys[0])
                for c in conflict_by_rel[rel]
            }
            candidates = [
                r
                for r in clean_rows
                if project_row(r, schema.candidate_keys[0], schema.attr_to_idx)
                not in conflict_keys
            ]
            if not candidates:
                candidates = list(clean_rows)
            clean_row = candidates[(q_index * 17 + seed) % len(candidates)]
            key_vals = {a: clean_row[schema.attr_to_idx[a]] for a in key_attrs}
            dirty_row = None

        answer = clean_row[schema.attr_to_idx[answer_attr]]
        question_text = tmpl["template"].format(**key_vals)

        clean_target = same_field_order([row_to_dict(schema, clean_row)], field_order)[0]
        pool_dicts = _rows_as_dicts(schema, clean_rows)
        exclude = {tuple(key_vals[a] for a in key_attrs)}
        # For enrollment query key may equal candidate key; for student/course same.

        if category == "No-Conflict":
            distractors = sample_distractors(
                pool_dicts,
                exclude_keys=exclude,
                key_attrs=key_attrs,
                question_id=qid,
                seed=evidence_order_seed,
            )
            clean_evidence = same_field_order([clean_target] + distractors, field_order)
            dirty_evidence = list(clean_evidence)
            valid_repair_evidence = list(clean_evidence)
            assert not _evidence_has_violation(schema, dirty_evidence)
            assert not _target_key_conflicted(schema, dirty_evidence, key_vals)

        elif category == "Irrelevant-Conflict":
            # Pick an irrelevant conflict pair (different key).
            confs = [
                c
                for c in conflict_by_rel[rel]
                if any(c["key"].get(a) != key_vals.get(a) for a in schema.candidate_keys[0])
            ]
            if not confs:
                # Synthesize an irrelevant conflict on another clean row.
                other = clean_rows[(q_index * 31 + 7) % len(clean_rows)]
                while project_row(other, key_attrs, schema.attr_to_idx) == tuple(
                    key_vals[a] for a in key_attrs
                ):
                    other = clean_rows[(hash(str(other)) + 3) % len(clean_rows)]
                irr_dirty, _ = make_conflicting_row(
                    schema, other, random.Random(f"{seed}:{qid}:irr"), f"IRR-{qid}"
                )
                irr_clean_d = row_to_dict(schema, other)
                irr_dirty_d = row_to_dict(schema, irr_dirty)
            else:
                conf = confs[(q_index - 1) % len(confs)]
                irr_clean_d = conf["clean_row"]
                irr_dirty_d = conf["conflicting_row"]

            irr_pair = shuffle_pair(
                same_field_order([irr_clean_d], field_order)[0],
                same_field_order([irr_dirty_d], field_order)[0],
                qid,
                evidence_order_seed,
            )
            distractors = sample_distractors(
                pool_dicts,
                exclude_keys=exclude
                | {tuple(irr_clean_d[a] for a in key_attrs)},
                key_attrs=key_attrs,
                question_id=qid,
                seed=evidence_order_seed,
                k_min=1,
                k_max=2,
            )
            dirty_evidence = same_field_order(
                [clean_target] + irr_pair + distractors, field_order
            )
            clean_evidence = same_field_order(
                [clean_target] + distractors, field_order
            )
            valid_repair_evidence = same_field_order(
                [clean_target, irr_clean_d] + distractors, field_order
            )
            assert _evidence_has_violation(schema, dirty_evidence)
            assert not _target_key_conflicted(schema, dirty_evidence, key_vals)

        else:  # Answer-Critical-Conflict
            assert dirty_row is not None
            pair = shuffle_pair(
                same_field_order([row_to_dict(schema, clean_row)], field_order)[0],
                same_field_order([row_to_dict(schema, dirty_row)], field_order)[0],
                qid,
                evidence_order_seed,
            )
            distractors = sample_distractors(
                pool_dicts,
                exclude_keys=exclude,
                key_attrs=key_attrs,
                question_id=qid,
                seed=evidence_order_seed,
                k_min=1,
                k_max=2,
            )
            dirty_evidence = same_field_order(pair + distractors, field_order)
            clean_evidence = same_field_order([clean_target] + distractors, field_order)
            valid_repair_evidence = list(clean_evidence)
            assert _evidence_has_violation(schema, dirty_evidence)
            assert _target_key_conflicted(schema, dirty_evidence, key_vals)

        # Sanity: clean/valid_repair evidence never violate F
        assert not _evidence_has_violation(schema, clean_evidence)
        assert not _evidence_has_violation(schema, valid_repair_evidence)
        assert satisfies_fds(
            [tuple(r[a] for a in field_order) for r in clean_evidence],
            schema.fds,
            schema.attr_to_idx,
        )

        q_obj = {
            "question_id": qid,
            "question": question_text,
            "relation": rel,
            "query_key": key_vals,
            "answer_attr": answer_attr,
            "answer": answer,
            "category": category,
            "expected_status": {
                "clean": "ANSWER",
                "dirty": "CONFLICT" if category == "Answer-Critical-Conflict" else "ANSWER",
                "valid_repair": "ANSWER",
                "dirty_fd": "CONFLICT" if category == "Answer-Critical-Conflict" else "ANSWER",
            },
            "clean_source_label": "clean_oracle",
            "evidence_order_seed": evidence_order_seed,
        }
        e_obj = {
            "question_id": qid,
            "relation": rel,
            "field_order": list(field_order),
            "clean_evidence": clean_evidence,
            "dirty_evidence": dirty_evidence,
            "valid_repair_evidence": valid_repair_evidence,
        }
        questions.append(q_obj)
        evidence_rows.append(e_obj)

    return questions, evidence_rows
