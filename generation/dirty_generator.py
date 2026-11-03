"""Inject BCNF FD conflicts into clean relations."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from generation.database_generator import (
    dict_to_row,
    row_to_dict,
    save_relation_csv,
    save_relation_jsonl,
)
from generation.schemas import get_schema
from utils.bootstrap import ensure_experiments_on_path
from utils.io_utils import ensure_dir, write_json

ensure_experiments_on_path()

from common.fd_utils import find_fd_violation, project_row, satisfies_fds  # noqa: E402
from common.types import RelationSchema, Row  # noqa: E402

# Alternate values used when mutating non-key attributes.
ALT_MAJORS = (
    "Computer Science",
    "Mathematics",
    "Physics",
    "Economics",
    "Mechanical Engineering",
    "Electronic Engineering",
)
ALT_GRADES = ("1", "2", "3", "4")
ALT_DEPARTMENTS = (
    "Computer Science",
    "Mathematics",
    "Physics",
    "Economics",
    "Engineering",
)
ALT_COURSE_STEMS = (
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
)


def _alt_value(attr: str, current: Any, rng: random.Random, tag: int) -> Any:
    if attr == "name":
        return f"Student_Conflict_{tag}"
    if attr == "major":
        choices = [m for m in ALT_MAJORS if m != current]
        return rng.choice(choices)
    if attr == "grade_level":
        choices = [g for g in ALT_GRADES if g != str(current)]
        return rng.choice(choices)
    if attr == "course_name":
        stem = rng.choice(ALT_COURSE_STEMS)
        return f"{stem} Conflict {tag}"
    if attr == "credits":
        choices = [c for c in range(1, 6) if c != int(current)]
        return rng.choice(choices)
    if attr == "department":
        choices = [d for d in ALT_DEPARTMENTS if d != current]
        return rng.choice(choices)
    if attr == "score":
        choices = [s for s in range(0, 101) if s != int(current)]
        return rng.choice(choices)
    return f"{current}__conflict_{tag}"


def make_conflicting_row(
    schema: RelationSchema,
    clean_row: Row,
    rng: random.Random,
    conflict_id: str,
    change_attrs: list[str] | None = None,
) -> tuple[Row, list[str]]:
    """Keep candidate-key attrs; mutate at least one non-key attribute."""
    key_attrs = set(schema.candidate_keys[0])
    non_key = [a for a in schema.attributes if a not in key_attrs]
    if not non_key:
        raise ValueError(f"No non-key attributes in {schema.name}")
    if change_attrs is None:
        n_change = 1 if len(non_key) == 1 else rng.randint(1, min(2, len(non_key)))
        change_attrs = rng.sample(non_key, n_change)
    values = list(clean_row)
    tag = abs(hash(conflict_id)) % (10**6)
    for attr in change_attrs:
        idx = schema.attr_to_idx[attr]
        values[idx] = _alt_value(attr, values[idx], rng, tag)
    dirty = tuple(values)
    if dirty == clean_row:
        # Force a change if RNG collided.
        attr = change_attrs[0]
        idx = schema.attr_to_idx[attr]
        values[idx] = _alt_value(attr, values[idx], rng, tag + 1)
        dirty = tuple(values)
    assert dirty != clean_row
    assert project_row(dirty, schema.candidate_keys[0], schema.attr_to_idx) == project_row(
        clean_row, schema.candidate_keys[0], schema.attr_to_idx
    )
    return dirty, list(change_attrs)


def key_dict(schema: RelationSchema, row: Row) -> dict[str, Any]:
    key = schema.candidate_keys[0]
    return {a: row[schema.attr_to_idx[a]] for a in key}


def generate_dirty_database(
    clean_db: dict[str, Any],
    seed: int = 42,
    conflicts_per_relation: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Copy clean DB and inject FD conflicts. Returns dirty rows + audit log."""
    rng = random.Random(seed)
    if conflicts_per_relation is None:
        # Enough conflicts for answer-critical and irrelevant categories.
        conflicts_per_relation = {
            "STUDENT": 200,
            "COURSE": 40,
            "ENROLLMENT": 400,
        }

    dirty_db: dict[str, Any] = {"meta": {"seed": seed, "conflicts_per_relation": conflicts_per_relation}}
    conflict_records: list[dict[str, Any]] = []
    conflict_seq = 0

    for rel_name in ("STUDENT", "COURSE", "ENROLLMENT"):
        schema: RelationSchema = clean_db[rel_name]["schema"]
        clean_rows: list[Row] = list(clean_db[rel_name]["rows"])
        dirty_rows = list(clean_rows)
        n_conf = min(conflicts_per_relation[rel_name], len(clean_rows))
        chosen = rng.sample(range(len(clean_rows)), n_conf)
        for idx in chosen:
            conflict_seq += 1
            conflict_id = f"CF{conflict_seq:06d}"
            clean_row = clean_rows[idx]
            dirty_row, changed = make_conflicting_row(schema, clean_row, rng, conflict_id)
            dirty_rows.append(dirty_row)
            conflict_records.append(
                {
                    "conflict_id": conflict_id,
                    "relation": rel_name,
                    "key": key_dict(schema, clean_row),
                    "clean_row": row_to_dict(schema, clean_row),
                    "conflicting_row": row_to_dict(schema, dirty_row),
                    "changed_attributes": changed,
                }
            )
        assert not satisfies_fds(dirty_rows, schema.fds, schema.attr_to_idx)
        assert find_fd_violation(dirty_rows, schema.fds, schema.attr_to_idx) is not None
        dirty_db[rel_name] = {"schema": schema, "rows": dirty_rows}

    dirty_db["conflicts"] = conflict_records
    dirty_db["meta"]["conflict_count"] = len(conflict_records)
    return dirty_db


def save_dirty_database(dirty_db: dict[str, Any], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
        schema: RelationSchema = dirty_db[rel]["schema"]
        rows: list[Row] = dirty_db[rel]["rows"]
        save_relation_csv(out_dir / f"{rel.lower()}.csv", schema, rows)
        save_relation_jsonl(out_dir / f"{rel.lower()}.jsonl", schema, rows)
    write_json(out_dir / "conflicts.json", dirty_db["conflicts"])
    write_json(out_dir / "meta.json", dirty_db["meta"])


def load_dirty_database(dirty_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "conflicts": json.loads((dirty_dir / "conflicts.json").read_text(encoding="utf-8")),
        "meta": json.loads((dirty_dir / "meta.json").read_text(encoding="utf-8")),
    }
    for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
        schema = get_schema(rel)
        rows = []
        with (dirty_dir / f"{rel.lower()}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(dict_to_row(schema, json.loads(line)))
        out[rel] = {"schema": schema, "rows": rows}
    return out
