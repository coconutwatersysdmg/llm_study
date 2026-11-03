"""Schema definitions for the controlled BCNF LLM study."""

from __future__ import annotations

from utils.bootstrap import ensure_experiments_on_path

ensure_experiments_on_path()

from common.fd_utils import validate_bcnf  # noqa: E402
from common.types import FD, RelationSchema  # noqa: E402


def student_schema() -> RelationSchema:
    s = RelationSchema(
        attributes=("student_id", "name", "major", "grade_level"),
        fds=(FD(("student_id",), ("name", "major", "grade_level")),),
        candidate_keys=(("student_id",),),
        name="STUDENT",
    )
    if not validate_bcnf(s.U, s.fds):
        raise ValueError("STUDENT schema failed BCNF validation")
    s.validate_bcnf()
    return s


def course_schema() -> RelationSchema:
    s = RelationSchema(
        attributes=("course_id", "course_name", "credits", "department"),
        fds=(FD(("course_id",), ("course_name", "credits", "department")),),
        candidate_keys=(("course_id",),),
        name="COURSE",
    )
    if not validate_bcnf(s.U, s.fds):
        raise ValueError("COURSE schema failed BCNF validation")
    s.validate_bcnf()
    return s


def enrollment_schema() -> RelationSchema:
    s = RelationSchema(
        attributes=("student_id", "course_id", "score"),
        fds=(FD(("student_id", "course_id"), ("score",)),),
        candidate_keys=(("student_id", "course_id"),),
        name="ENROLLMENT",
    )
    if not validate_bcnf(s.U, s.fds):
        raise ValueError("ENROLLMENT schema failed BCNF validation")
    s.validate_bcnf()
    return s


SCHEMAS = {
    "STUDENT": student_schema,
    "COURSE": course_schema,
    "ENROLLMENT": enrollment_schema,
}


def get_schema(name: str) -> RelationSchema:
    return SCHEMAS[name]()


def fd_strings(schema: RelationSchema) -> list[str]:
    return [str(fd) for fd in schema.fds]
