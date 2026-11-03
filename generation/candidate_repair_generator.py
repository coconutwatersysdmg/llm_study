"""Candidate repair generators for Experiment 2 (over-deletion / residual)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from checker.repair_checker_adapter import (
    AdapterCheckResult,
    complete_overdeleted_candidate,
    rows_from_dicts,
    verify_valid_repair,
)
from generation.database_generator import row_to_dict
from generation.dirty_generator import make_conflicting_row
from generation.schemas import get_schema
from utils.bootstrap import ensure_experiments_on_path
from utils.io_utils import ensure_dir, write_json

ensure_experiments_on_path()

from common.types import RelationSchema, Row  # noqa: E402


def build_valid_repairs_from_clean(clean_db: dict[str, Any]) -> dict[str, Any]:
    """Use clean relations as known-legal S-repair candidates, then verify via checker.

    The clean oracle is used ONLY for experimental construction.
    It is NEVER passed into the checker as ground-truth labels.
    """
    out: dict[str, Any] = {"relations": {}, "verification": {}}
    for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
        schema: RelationSchema = clean_db[rel]["schema"]
        clean_rows: list[Row] = list(clean_db[rel]["rows"])
        # For a consistent clean instance, r' = r is a trivial S-repair of itself.
        # Experimentally we also treat clean as repair of dirty by using dirty as r.
        result = verify_valid_repair(schema, clean_rows, clean_rows)
        if not (result.candidate_consistent and result.is_repair):
            raise RuntimeError(
                f"Clean self-repair failed for {rel}: "
                f"consistent={result.candidate_consistent}, is_repair={result.is_repair}"
            )
        out["relations"][rel] = {
            "schema": schema.to_json(),
            "rows": [row_to_dict(schema, r) for r in clean_rows],
        }
        out["verification"][rel] = {
            "candidate_consistent": result.candidate_consistent,
            "is_repair": result.is_repair,
            "note": (
                "Clean oracle used only for construction. "
                "Checker validates consistency + maximality only; "
                "it does not decide real-world truth."
            ),
        }
    return out


def build_valid_repair_of_dirty(
    clean_db: dict[str, Any],
    dirty_db: dict[str, Any],
) -> dict[str, Any]:
    """Validate clean ⊂ dirty as an S-repair of dirty via BCNFRepairChecker."""
    out: dict[str, Any] = {"relations": {}, "verification": {}}
    for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
        schema: RelationSchema = clean_db[rel]["schema"]
        clean_rows: list[Row] = list(clean_db[rel]["rows"])
        dirty_rows: list[Row] = list(dirty_db[rel]["rows"])
        # Ensure clean ⊆ dirty
        if not set(clean_rows) <= set(dirty_rows):
            raise RuntimeError(f"Clean rows are not a subset of dirty for {rel}")
        result = verify_valid_repair(schema, dirty_rows, clean_rows)
        if not (result.candidate_consistent and result.is_repair):
            raise RuntimeError(
                f"Valid repair check failed for {rel}: "
                f"consistent={result.candidate_consistent}, is_repair={result.is_repair}"
            )
        out["relations"][rel] = {
            "schema": schema.to_json(),
            "rows": [row_to_dict(schema, r) for r in clean_rows],
        }
        out["verification"][rel] = {
            "candidate_consistent": True,
            "is_repair": True,
            "r_size": len(dirty_rows),
            "r_prime_size": len(clean_rows),
            "boundary": (
                "Checker confirms F-satisfaction and S-repair maximality. "
                "It does NOT claim retained tuples are real-world facts."
            ),
        }
    return out


def save_valid_repair(valid: dict[str, Any], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for rel, payload in valid["relations"].items():
        path = out_dir / f"{rel.lower()}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in payload["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(out_dir / f"{rel.lower()}_schema.json", payload["schema"])
    write_json(out_dir / "verification.json", valid["verification"])


def _select_answer_critical_keys(
    questions: list[dict[str, Any]],
    relation: str,
) -> list[dict[str, Any]]:
    keys = []
    seen = set()
    for q in questions:
        if q["relation"] != relation:
            continue
        if q["category"] != "Answer-Critical-Conflict":
            continue
        key_t = tuple(sorted(q["query_key"].items()))
        if key_t in seen:
            continue
        seen.add(key_t)
        keys.append(q["query_key"])
    return keys


def generate_overdeletion_candidates(
    dirty_db: dict[str, Any],
    valid_repair: dict[str, Any],
    questions: list[dict[str, Any]],
    error_rates: list[float],
    seed: int = 42,
) -> dict[str, Any]:
    """Create over-deleted candidates that remain consistent but non-maximal."""
    rng = random.Random(seed)
    artifacts: dict[str, Any] = {"error_rates": {}, "seed": seed}

    for rate in error_rates:
        rate_key = f"{rate:.2f}"
        artifacts["error_rates"][rate_key] = {}
        for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
            schema = get_schema(rel)
            dirty_rows = list(dirty_db[rel]["rows"])
            valid_rows = rows_from_dicts(schema, valid_repair["relations"][rel]["rows"])
            valid_set = list(valid_rows)

            # Prefer deleting tuples that affect answer-critical questions.
            critical_keys = _select_answer_critical_keys(questions, rel)
            key_attrs = schema.candidate_keys[0]

            def matches(row: Row, key: dict[str, Any]) -> bool:
                return all(row[schema.attr_to_idx[a]] == key[a] for a in key_attrs if a in key)

            preferred = []
            for key in critical_keys:
                for row in valid_set:
                    if matches(row, key):
                        preferred.append(row)
            preferred = list(dict.fromkeys(preferred))  # stable unique
            other = [r for r in valid_set if r not in preferred]
            deletable = preferred + other
            n_del = max(1, int(round(len(deletable) * rate))) if deletable else 0
            n_del = min(n_del, max(0, len(valid_set) - 1))
            to_delete = deletable[:n_del]
            candidate = [r for r in valid_set if r not in set(to_delete)]

            from checker.repair_checker_adapter import check_candidate

            pre = check_candidate(schema, dirty_rows, candidate)
            if not pre.candidate_consistent:
                raise RuntimeError(f"Over-deletion broke consistency at rate={rate}, rel={rel}")
            if pre.is_repair:
                raise RuntimeError(
                    f"Over-deletion still maximal at rate={rate}, rel={rel}; increase deletions"
                )

            completed = complete_overdeleted_candidate(schema, dirty_rows, candidate)
            artifacts["error_rates"][rate_key][rel] = {
                "error_rate": rate,
                "deleted_count": len(to_delete),
                "deleted_rows": [row_to_dict(schema, r) for r in to_delete],
                "initial_candidate": [row_to_dict(schema, r) for r in candidate],
                "initial_candidate_size": len(candidate),
                "initial_check": {
                    "candidate_consistent": pre.candidate_consistent,
                    "is_repair": pre.is_repair,
                    "addable_tuple": list(pre.addable_tuple) if pre.addable_tuple else None,
                },
                "completed_candidate": [row_to_dict(schema, r) for r in completed["r_prime"]],
                "final_candidate_size": completed["final_candidate_size"],
                "checker_iterations": completed["checker_iterations"],
                "added_back_count": completed["added_back_count"],
                "final_is_repair": completed["final_is_repair"],
                "final_candidate_consistent": completed["candidate_consistent"],
            }
    return artifacts


def generate_residual_candidates(
    dirty_db: dict[str, Any],
    valid_repair: dict[str, Any],
    error_rates: list[float],
    seed: int = 42,
) -> dict[str, Any]:
    """Re-introduce conflicting tuples into a valid repair ⇒ inconsistent candidate.

    Checker must detect inconsistency. It must NOT choose which conflicting
    tuple is the real-world truth.
    """
    rng = random.Random(seed + 99)
    artifacts: dict[str, Any] = {"error_rates": {}, "seed": seed}

    for rate in error_rates:
        rate_key = f"{rate:.2f}"
        artifacts["error_rates"][rate_key] = {}
        for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
            schema = get_schema(rel)
            dirty_rows = list(dirty_db[rel]["rows"])
            valid_rows = rows_from_dicts(schema, valid_repair["relations"][rel]["rows"])
            n_inj = max(1, int(round(len(valid_rows) * rate)))
            n_inj = min(n_inj, len(valid_rows))
            targets = rng.sample(valid_rows, n_inj)
            injected: list[Row] = []
            candidate = list(valid_rows)
            universe = set(dirty_rows)
            for i, s in enumerate(targets):
                twin, _ = make_conflicting_row(
                    schema, s, rng, f"RES-{rel}-{rate_key}-{i}"
                )
                candidate.append(twin)
                universe.add(twin)
                injected.append(twin)

            from checker.repair_checker_adapter import check_candidate

            res = check_candidate(schema, list(universe), candidate)
            if res.candidate_consistent:
                raise RuntimeError(
                    f"Residual candidate unexpectedly consistent at rate={rate}, rel={rel}"
                )

            artifacts["error_rates"][rate_key][rel] = {
                "error_rate": rate,
                "injected_count": len(injected),
                "injected_rows": [row_to_dict(schema, r) for r in injected],
                "candidate": [row_to_dict(schema, r) for r in candidate],
                "universe": [row_to_dict(schema, r) for r in universe],
                "candidate_consistent": res.candidate_consistent,
                "is_repair": res.is_repair,
                "checker_rejected": True,
                "boundary": (
                    "Detection only. Checker does not select which conflicting "
                    "tuple to keep."
                ),
            }
    return artifacts


def save_candidate_artifacts(artifacts: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    write_json(path, artifacts)
