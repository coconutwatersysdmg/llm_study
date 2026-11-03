"""Deterministic clean BCNF university database generator."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from generation.schemas import course_schema, enrollment_schema, student_schema
from utils.bootstrap import ensure_experiments_on_path
from utils.io_utils import ensure_dir, write_json

ensure_experiments_on_path()

from common.fd_utils import satisfies_fds  # noqa: E402
from common.types import RelationSchema, Row  # noqa: E402

MAJORS = (
    "Computer Science",
    "Mathematics",
    "Physics",
    "Economics",
    "Mechanical Engineering",
    "Electronic Engineering",
)
GRADE_LEVELS = ("1", "2", "3", "4")
DEPARTMENTS = (
    "Computer Science",
    "Mathematics",
    "Physics",
    "Economics",
    "Engineering",
)
COURSE_NAME_STEMS = (
    "Intro Programming",
    "Data Structures",
    "Databases",
    "Algorithms",
    "Linear Algebra",
    "Probability",
    "Operating Systems",
    "Computer Networks",
    "AI Foundations",
    "Discrete Math",
    "Calculus",
    "Statistics",
)


def row_to_dict(schema: RelationSchema, row: Row) -> dict[str, Any]:
    return {a: row[i] for i, a in enumerate(schema.attributes)}


def dict_to_row(schema: RelationSchema, d: dict[str, Any]) -> Row:
    return tuple(d[a] for a in schema.attributes)


def generate_clean_database(
    n_students: int = 1000,
    n_courses: int = 100,
    n_enrollments: int = 15000,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate clean STUDENT / COURSE / ENROLLMENT satisfying F."""
    stud_s = student_schema()
    cour_s = course_schema()
    enr_s = enrollment_schema()

    students: list[Row] = []
    for i in range(n_students):
        students.append(
            (
                f"S{i:04d}",
                f"Student_{i:04d}",
                MAJORS[i % len(MAJORS)],
                GRADE_LEVELS[i % len(GRADE_LEVELS)],
            )
        )

    courses: list[Row] = []
    for j in range(n_courses):
        courses.append(
            (
                f"C{j:03d}",
                f"{COURSE_NAME_STEMS[j % len(COURSE_NAME_STEMS)]} {j:03d}",
                1 + (j % 5),
                DEPARTMENTS[j % len(DEPARTMENTS)],
            )
        )

    enrollments: list[Row] = []
    seen: set[tuple[int, int]] = set()
    target = min(n_enrollments, n_students * min(8, n_courses))
    for i in range(n_students):
        n_take = 3 + (i % 6)
        for t in range(n_take):
            j = (i * 7 + t * 13 + seed) % n_courses
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            score = (i * 17 + j * 31 + seed) % 101
            enrollments.append((f"S{i:04d}", f"C{j:03d}", score))
            if len(enrollments) >= target:
                break
        if len(enrollments) >= target:
            break

    assert satisfies_fds(students, stud_s.fds, stud_s.attr_to_idx)
    assert satisfies_fds(courses, cour_s.fds, cour_s.attr_to_idx)
    assert satisfies_fds(enrollments, enr_s.fds, enr_s.attr_to_idx)

    return {
        "STUDENT": {"schema": stud_s, "rows": students},
        "COURSE": {"schema": cour_s, "rows": courses},
        "ENROLLMENT": {"schema": enr_s, "rows": enrollments},
        "meta": {
            "n_students": len(students),
            "n_courses": len(courses),
            "n_enrollments": len(enrollments),
            "seed": seed,
        },
    }


def save_relation_csv(path: Path, schema: RelationSchema, rows: list[Row]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(schema.attributes))
        w.writeheader()
        for row in rows:
            w.writerow(row_to_dict(schema, row))


def save_relation_jsonl(path: Path, schema: RelationSchema, rows: list[Row]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row_to_dict(schema, row), ensure_ascii=False) + "\n")


def save_clean_database(db: dict[str, Any], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
        schema: RelationSchema = db[rel]["schema"]
        rows: list[Row] = db[rel]["rows"]
        write_json(out_dir / f"{rel.lower()}_schema.json", schema.to_json())
        save_relation_csv(out_dir / f"{rel.lower()}.csv", schema, rows)
        save_relation_jsonl(out_dir / f"{rel.lower()}.jsonl", schema, rows)
    write_json(out_dir / "meta.json", db["meta"])


def load_relation_jsonl(path: Path, schema: RelationSchema) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(dict_to_row(schema, json.loads(line)))
    return rows


def load_clean_database(clean_dir: Path) -> dict[str, Any]:
    stud_s = student_schema()
    cour_s = course_schema()
    enr_s = enrollment_schema()
    return {
        "STUDENT": {
            "schema": stud_s,
            "rows": load_relation_jsonl(clean_dir / "student.jsonl", stud_s),
        },
        "COURSE": {
            "schema": cour_s,
            "rows": load_relation_jsonl(clean_dir / "course.jsonl", cour_s),
        },
        "ENROLLMENT": {
            "schema": enr_s,
            "rows": load_relation_jsonl(clean_dir / "enrollment.jsonl", enr_s),
        },
        "meta": json.loads((clean_dir / "meta.json").read_text(encoding="utf-8")),
    }
