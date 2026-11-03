"""Exp2A/2B candidate construction under formal experimental protocol.

Exp2A over-deletion ONLY targets tuples whose candidate key is unique in dirty r
(so addable_tuple restoration never requires choosing among conflicting facts).

Checker completion uses ONLY r, r', and addable_tuple — never GT / clean labels.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from checker.repair_checker_adapter import (
    check_candidate,
    complete_overdeleted_candidate,
    rows_from_dicts,
)
from generation.database_generator import row_to_dict
from generation.schemas import get_schema
from utils.bootstrap import ensure_experiments_on_path
from utils.io_utils import ensure_dir, write_json

ensure_experiments_on_path()

from common.types import RelationSchema, Row  # noqa: E402
from common.fd_utils import project_row  # noqa: E402


def _key_tuple(schema: RelationSchema, row: Row) -> tuple[Any, ...]:
    return project_row(row, schema.candidate_keys[0], schema.attr_to_idx)


def _key_from_dict(schema: RelationSchema, d: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(d[a] for a in schema.candidate_keys[0])


def build_key_multiplicity(dirty_rows: list[Row], schema: RelationSchema) -> dict[tuple[Any, ...], int]:
    counts: dict[tuple[Any, ...], int] = {}
    for row in dirty_rows:
        k = _key_tuple(schema, row)
        counts[k] = counts.get(k, 0) + 1
    return counts


def build_overdeletion_eligible_pool(
    dirty_db: dict[str, Any],
    valid_repair: dict[str, Any],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Eligible = in ValidRepair AND key unique in dirty AND is a QA target."""
    pool: dict[str, Any] = {"relations": {}, "total": 0}
    for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
        schema = get_schema(rel)
        dirty_rows = list(dirty_db[rel]["rows"])
        valid_rows = rows_from_dicts(schema, valid_repair["relations"][rel]["rows"])
        mult = build_key_multiplicity(dirty_rows, schema)

        qa_keys: set[tuple[Any, ...]] = set()
        qids_by_key: dict[tuple[Any, ...], list[str]] = {}
        for q in questions:
            if q["relation"] != rel:
                continue
            # Map query_key onto candidate-key attributes
            ck = schema.candidate_keys[0]
            if not all(a in q["query_key"] for a in ck):
                continue
            kt = tuple(q["query_key"][a] for a in ck)
            qa_keys.add(kt)
            qids_by_key.setdefault(kt, []).append(q["question_id"])

        entries = []
        for row in valid_rows:
            kt = _key_tuple(schema, row)
            if mult.get(kt, 0) != 1:
                continue  # key not unique in dirty → excluded
            if kt not in qa_keys:
                continue
            entries.append(
                {
                    "row": row_to_dict(schema, row),
                    "key": {a: row[schema.attr_to_idx[a]] for a in schema.candidate_keys[0]},
                    "question_ids": sorted(qids_by_key[kt]),
                }
            )
        pool["relations"][rel] = {
            "count": len(entries),
            "entries": entries,
        }
        pool["total"] += len(entries)
    return pool


def generate_overdeletion_protocol(
    dirty_db: dict[str, Any],
    valid_repair: dict[str, Any],
    questions: list[dict[str, Any]],
    error_rates: list[float],
    corruption_seeds: list[int],
) -> dict[str, Any]:
    """Build over-deletion candidates + question manifests for all rate×seed."""
    eligible = build_overdeletion_eligible_pool(dirty_db, valid_repair, questions)
    q_by_id = {q["question_id"]: q for q in questions}
    artifacts: dict[str, Any] = {
        "eligible_pool_total": eligible["total"],
        "error_rates": {},
        "protocol": (
            "Over-delete only unique-key ValidRepair tuples that are QA targets. "
            "Checker restores maximality via addable_tuple; does not decide truth."
        ),
    }
    question_manifest: dict[str, Any] = {"runs": []}

    for rate in error_rates:
        rate_key = f"{rate:.2f}"
        artifacts["error_rates"][rate_key] = {}
        for seed in corruption_seeds:
            rng = random.Random(seed)
            run_key = f"{rate_key}__seed{seed}"
            artifacts["error_rates"][rate_key][f"seed{seed}"] = {}
            affected_qids: set[str] = set()
            unaffected_qids: set[str] = set()
            run_meta: dict[str, Any] = {
                "error_rate": rate,
                "corruption_seed": seed,
                "relations": {},
            }

            for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
                schema = get_schema(rel)
                dirty_rows = list(dirty_db[rel]["rows"])
                valid_rows = rows_from_dicts(schema, valid_repair["relations"][rel]["rows"])
                entries = list(eligible["relations"][rel]["entries"])
                if not entries:
                    raise RuntimeError(f"Empty eligible pool for {rel}")

                n_del = max(1, int(round(len(entries) * rate)))
                n_del = min(n_del, len(entries))
                chosen = rng.sample(entries, n_del)
                del_rows = [tuple(e["row"][a] for a in schema.attributes) for e in chosen]
                del_set = set(del_rows)
                candidate = [r for r in valid_rows if r not in del_set]

                pre = check_candidate(schema, dirty_rows, candidate)
                if not pre.candidate_consistent or pre.is_repair:
                    raise RuntimeError(
                        f"OVERDELETION_CONSTRUCTION_ERROR rate={rate} seed={seed} rel={rel} "
                        f"consistent={pre.candidate_consistent} is_repair={pre.is_repair}"
                    )

                completed = complete_overdeleted_candidate(schema, dirty_rows, candidate)
                if not (completed["final_is_repair"] and completed["candidate_consistent"]):
                    raise RuntimeError(
                        f"OVERDELETION_COMPLETION_ERROR rate={rate} seed={seed} rel={rel}"
                    )

                # Protocol check via generator metadata (checker itself never saw this):
                restored = set(completed["r_prime"])
                for dr in del_rows:
                    if dr not in restored:
                        raise RuntimeError(
                            f"Unique-key overdeleted tuple not restored: {rel} {dr}"
                        )

                for e in chosen:
                    affected_qids.update(e["question_ids"])
                for e in entries:
                    if e not in chosen:
                        unaffected_qids.update(e["question_ids"])

                run_meta["relations"][rel] = {
                    "deleted_count": len(del_rows),
                    "deleted_rows": [row_to_dict(schema, r) for r in del_rows],
                    "deleted_question_ids": sorted(
                        {qid for e in chosen for qid in e["question_ids"]}
                    ),
                    "initial_candidate": [row_to_dict(schema, r) for r in candidate],
                    "completed_candidate": [
                        row_to_dict(schema, r) for r in completed["r_prime"]
                    ],
                    "initial_check": {
                        "candidate_consistent": True,
                        "is_repair": False,
                    },
                    "checker_iterations": completed["checker_iterations"],
                    "added_back_count": completed["added_back_count"],
                    "final_is_repair": True,
                    "final_candidate_consistent": True,
                }
                artifacts["error_rates"][rate_key][f"seed{seed}"][rel] = run_meta["relations"][rel]

            # Matched unaffected sample: same size as affected
            affected_sorted = sorted(affected_qids)
            unaff_pool = sorted(unaffected_qids - affected_qids)
            # Also allow any question whose target was not deleted
            for q in questions:
                if q["question_id"] in affected_qids:
                    continue
                unaff_pool.append(q["question_id"])
            unaff_pool = sorted(set(unaff_pool))
            ctrl_rng = random.Random(seed * 1009 + int(rate * 10000))
            n_ctrl = min(len(affected_sorted), len(unaff_pool))
            controls = sorted(ctrl_rng.sample(unaff_pool, n_ctrl)) if n_ctrl else []

            question_manifest["runs"].append(
                {
                    "error_rate": rate,
                    "corruption_seed": seed,
                    "run_key": run_key,
                    "affected_question_ids": affected_sorted,
                    "unaffected_control_question_ids": controls,
                    "all_eval_question_ids": sorted(set(affected_sorted) | set(controls)),
                }
            )

    artifacts["question_manifest"] = question_manifest
    artifacts["eligible_pool"] = {
        "total": eligible["total"],
        "counts": {rel: eligible["relations"][rel]["count"] for rel in eligible["relations"]},
    }
    return artifacts, eligible


def generate_residual_protocol(
    dirty_db: dict[str, Any],
    valid_repair: dict[str, Any],
    questions: list[dict[str, Any]],
    error_rates: list[float],
    corruption_seeds: list[int],
) -> dict[str, Any]:
    """Re-introduce dirty conflict twins into ValidRepair → inconsistent candidate."""
    artifacts: dict[str, Any] = {
        "error_rates": {},
        "protocol": (
            "Residual conflict: detect inconsistency only. "
            "Checker must NOT choose which conflicting tuple is true."
        ),
    }
    question_manifest: dict[str, Any] = {"runs": []}

    # Index dirty conflicts by relation+key
    dirty_by_key: dict[str, dict[tuple[Any, ...], list[Row]]] = {}
    for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
        schema = get_schema(rel)
        buckets: dict[tuple[Any, ...], list[Row]] = {}
        for row in dirty_db[rel]["rows"]:
            buckets.setdefault(_key_tuple(schema, row), []).append(row)
        dirty_by_key[rel] = buckets

    for rate in error_rates:
        rate_key = f"{rate:.2f}"
        artifacts["error_rates"][rate_key] = {}
        for seed in corruption_seeds:
            rng = random.Random(seed + 777)
            run_key = f"{rate_key}__seed{seed}"
            artifacts["error_rates"][rate_key][f"seed{seed}"] = {}
            affected_qids: set[str] = set()

            for rel in ("STUDENT", "COURSE", "ENROLLMENT"):
                schema = get_schema(rel)
                valid_rows = rows_from_dicts(schema, valid_repair["relations"][rel]["rows"])
                # Keys that have a conflicting twin in dirty (multiplicity >= 2)
                injectable = []
                for row in valid_rows:
                    kt = _key_tuple(schema, row)
                    twins = dirty_by_key[rel].get(kt, [])
                    extras = [t for t in twins if t != row]
                    if extras:
                        injectable.append((row, extras[0], kt))

                if not injectable:
                    raise RuntimeError(f"No residual-injectable keys for {rel}")

                n_inj = max(1, int(round(len(injectable) * rate)))
                n_inj = min(n_inj, len(injectable))
                chosen = rng.sample(injectable, n_inj)

                candidate = list(valid_rows)
                universe = set(dirty_db[rel]["rows"])
                injected = []
                injected_keys = []
                for retained, twin, kt in chosen:
                    candidate.append(twin)
                    universe.add(twin)
                    injected.append(row_to_dict(schema, twin))
                    injected_keys.append(kt)

                res = check_candidate(schema, list(universe), candidate)
                if res.candidate_consistent:
                    raise RuntimeError(
                        f"RESIDUAL_CONFLICT_CHECKER_FAILURE rate={rate} seed={seed} rel={rel}"
                    )

                # Map injected keys to Answer-Critical questions
                for q in questions:
                    if q["relation"] != rel:
                        continue
                    if q["category"] != "Answer-Critical-Conflict":
                        continue
                    ck = schema.candidate_keys[0]
                    if not all(a in q["query_key"] for a in ck):
                        continue
                    kt = tuple(q["query_key"][a] for a in ck)
                    if kt in {k for _, _, k in chosen}:
                        affected_qids.add(q["question_id"])

                artifacts["error_rates"][rate_key][f"seed{seed}"][rel] = {
                    "injected_count": len(injected),
                    "injected_rows": injected,
                    "candidate": [row_to_dict(schema, r) for r in candidate],
                    "universe": [row_to_dict(schema, r) for r in universe],
                    "candidate_consistent": False,
                    "is_repair": False,
                    "checker_rejected": True,
                }

            affected_sorted = sorted(affected_qids)
            # Unaffected matched controls from non-affected Answer-Critical or general
            pool = [
                q["question_id"]
                for q in questions
                if q["question_id"] not in affected_qids
            ]
            ctrl_rng = random.Random(seed * 1301 + int(rate * 10000))
            n_ctrl = min(len(affected_sorted), len(pool))
            controls = sorted(ctrl_rng.sample(pool, n_ctrl)) if n_ctrl else []
            question_manifest["runs"].append(
                {
                    "error_rate": rate,
                    "corruption_seed": seed,
                    "run_key": run_key,
                    "affected_question_ids": affected_sorted,
                    "unaffected_control_question_ids": controls,
                    "all_eval_question_ids": sorted(set(affected_sorted) | set(controls)),
                }
            )

    artifacts["question_manifest"] = question_manifest
    return artifacts


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    write_json(path, obj)
