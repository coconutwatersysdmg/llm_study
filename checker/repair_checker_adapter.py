"""Repair-checker adapter: import-only wrapper around frozen BCNFRepairChecker.

Capability boundary (must be preserved in code, README, and result interpretation):

  CAN verify:
    1) Candidate repair satisfies F (candidate_consistent)
    2) Candidate is inclusion-maximal S-repair (is_repair / addable_tuple)

  CANNOT:
    - decide which conflicting tuple is real-world true
    - claim deleted tuples are erroneous facts
    - claim retained tuples are true facts

  For over-deletion: may restore maximality via addable_tuple.
  For residual conflict: may detect inconsistency only; never auto-choose survivors.
"""

from __future__ import annotations

from typing import Any, Sequence

from utils.bootstrap import ensure_experiments_on_path

ensure_experiments_on_path()

from algorithms.bcnf_index import BCNFRepairChecker  # noqa: E402
from common.types import RelationSchema, RepairCheckResult, Row  # noqa: E402


class AdapterCheckResult:
    """Thin mirror of RepairCheckResult with study-friendly fields."""

    def __init__(self, raw: RepairCheckResult) -> None:
        self.raw = raw
        self.candidate_consistent = raw.candidate_consistent
        self.is_repair = raw.is_repair
        self.addable_tuple = raw.addable_tuple

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_consistent": self.candidate_consistent,
            "is_repair": self.is_repair,
            "addable_tuple": list(self.addable_tuple) if self.addable_tuple else None,
            "boundary": (
                "Checker validates F-satisfaction and S-repair maximality only. "
                "It does not decide real-world truth among conflicting tuples."
            ),
        }


def check_candidate(
    schema: RelationSchema,
    r: Sequence[Row],
    r_prime: Sequence[Row],
) -> AdapterCheckResult:
    checker = BCNFRepairChecker(schema, use_key_cover=False)
    return AdapterCheckResult(checker.check(list(r), list(r_prime)))


def verify_valid_repair(
    schema: RelationSchema,
    r: Sequence[Row],
    r_prime: Sequence[Row],
) -> AdapterCheckResult:
    return check_candidate(schema, r, r_prime)


def complete_overdeleted_candidate(
    schema: RelationSchema,
    r: Sequence[Row],
    r_prime: Sequence[Row],
    max_iters: int = 100000,
) -> dict[str, Any]:
    """Restore maximality using ONLY r, r', and checker.addable_tuple.

    Must NOT read clean answers, question ground truth, or source labels.
    Must NOT resolve residual conflicts (raises if candidate inconsistent).
    """
    checker = BCNFRepairChecker(schema, use_key_cover=False)
    candidate: set[Row] = set(r_prime)
    universe = list(r)
    added: list[Row] = []
    iterations = 0
    last: RepairCheckResult | None = None

    while True:
        iterations += 1
        if iterations > max_iters:
            raise RuntimeError("complete_overdeleted_candidate exceeded max_iters")
        last = checker.check(universe, list(candidate))
        if not last.candidate_consistent:
            raise ValueError(
                "Candidate is inconsistent (residual conflict). "
                "Completion must not auto-choose among conflicting tuples."
            )
        if last.is_repair:
            break
        if last.addable_tuple is None:
            raise RuntimeError("Non-maximal consistent candidate but no addable_tuple")
        candidate.add(last.addable_tuple)
        added.append(last.addable_tuple)

    return {
        "r_prime": list(candidate),
        "final_is_repair": True,
        "candidate_consistent": True,
        "checker_iterations": iterations,
        "added_back_count": len(added),
        "added_back": added,
        "initial_candidate_size": len(set(r_prime)),
        "final_candidate_size": len(candidate),
    }


def rows_from_dicts(schema: RelationSchema, dicts: Sequence[dict[str, Any]]) -> list[Row]:
    return [tuple(d[a] for a in schema.attributes) for d in dicts]
