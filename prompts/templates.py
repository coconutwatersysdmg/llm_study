"""Prompt templates. PROMPT_VERSION must bump on any formal-run change."""

from __future__ import annotations

from typing import Any

from generation.evidence_builder import format_evidence_block
from generation.schemas import fd_strings, get_schema

PROMPT_VERSION = "v1"

# Evaluator-only fields that must NEVER appear in prompts.
FORBIDDEN_PROMPT_FIELDS = (
    "answer",
    "clean_source_label",
    "expected_status",
    "ground_truth",
    "correct",
)

QA_SYSTEM = (
    "你是一个结构化数据库问答助手。\n"
    "只能依据给出的数据库记录回答问题。\n"
    "不要使用外部知识。\n"
    "如果这些记录不能支持唯一答案，只输出 UNKNOWN。\n"
    "如果能够确定答案，只输出最终答案本身。\n"
    "不要解释，不要添加其他文字。"
)

CONFLICT_SYSTEM = (
    "你是一个结构化数据库一致性检查与问答助手。\n"
    "只能依据给出的数据库记录和函数依赖进行判断。\n"
    "\n"
    "如果给出的记录违反指定函数依赖，只输出：\n"
    "\n"
    "CONFLICT\n"
    "\n"
    "如果记录不违反函数依赖并且能够唯一回答问题，只输出最终答案。\n"
    "\n"
    "如果记录不违反函数依赖但无法回答问题，只输出：\n"
    "\n"
    "UNKNOWN\n"
    "\n"
    "不要解释。"
)


def build_qa_user_prompt(evidence: list[dict[str, Any]], question: str) -> str:
    block = format_evidence_block(evidence)
    return f"数据库记录：\n{block}\n\n问题：\n{question}\n\n回答："


def build_conflict_user_prompt(
    relation: str,
    evidence: list[dict[str, Any]],
    question: str,
) -> str:
    schema = get_schema(relation)
    schema_line = f"{schema.name}(" + ", ".join(schema.attributes) + ")"
    fd_line = "; ".join(fd_strings(schema))
    block = format_evidence_block(evidence)
    return (
        f"关系模式：\n{schema_line}\n\n"
        f"函数依赖：\n{fd_line}\n\n"
        f"数据库记录：\n{block}\n\n"
        f"问题：\n{question}\n\n"
        f"回答："
    )


class PromptBuilder:
    """Builds prompts without reading evaluator-only metadata."""

    def __init__(self, version: str = PROMPT_VERSION) -> None:
        self.version = version

    def qa_prompts(
        self,
        question_text: str,
        evidence: list[dict[str, Any]],
    ) -> tuple[str, str]:
        return QA_SYSTEM, build_qa_user_prompt(evidence, question_text)

    def conflict_prompts(
        self,
        relation: str,
        question_text: str,
        evidence: list[dict[str, Any]],
    ) -> tuple[str, str]:
        return CONFLICT_SYSTEM, build_conflict_user_prompt(relation, evidence, question_text)

    def assert_no_leakage(self, system_prompt: str, user_prompt: str) -> None:
        blob = system_prompt + "\n" + user_prompt
        for field in FORBIDDEN_PROMPT_FIELDS:
            if f'"{field}"' in blob or f"'{field}'" in blob:
                raise AssertionError(f"Ground-truth field leaked into prompt: {field}")
