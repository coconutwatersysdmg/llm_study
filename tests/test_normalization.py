"""Normalizer and resume/cache tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clients.base import LLMClient, LLMResponse
from evaluation.normalize import answers_equal, normalize_answer
from evaluation.runner import load_success_ids, run_requests
from utils.io_utils import append_jsonl, ensure_dir


def test_normalizer() -> None:
    assert normalize_answer("  Computer Science  ")[0] == "Computer Science"
    assert normalize_answer("Answer: 85")[0] == "85"
    assert normalize_answer("答案：85")[0] == "85"
    assert normalize_answer("UNKNOWN")[0] == "UNKNOWN"
    tok, viol = normalize_answer("CONFLICT because duplicate key")
    assert tok == "CONFLICT"
    assert viol is True
    assert answers_equal("85", 85)
    assert answers_equal("85.0", 85)
    assert answers_equal("Computer Science", "computer science")
    assert not answers_equal("Mathematics", "Computer Science")


def test_resume_skips_successful_requests(tmp_path: Path) -> None:
    out = tmp_path / "raw.jsonl"
    ensure_dir(out.parent)
    append_jsonl(
        out,
        {
            "request_id": "exp1__glm__clean__Q000001",
            "success": True,
            "raw_response": "x",
        },
    )
    done = load_success_ids(out)
    assert "exp1__glm__clean__Q000001" in done

    calls = {"n": 0}

    class Dummy(LLMClient):
        provider = "zhipu"
        model = "dummy"

        def generate(self, system_prompt, user_prompt, request_id):
            calls["n"] += 1
            return LLMResponse(
                request_id=request_id,
                provider=self.provider,
                model=self.model,
                raw_text="ok",
                latency_sec=0.01,
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                success=True,
            )

    jobs = [
        {
            "request_id": "exp1__glm__clean__Q000001",
            "system_prompt": "s",
            "user_prompt": "u",
            "ground_truth": "ok",
            "experiment": "t",
            "condition": "clean",
            "question_id": "Q000001",
            "question_category": "No-Conflict",
        },
        {
            "request_id": "exp1__glm__clean__Q000002",
            "system_prompt": "s",
            "user_prompt": "u",
            "ground_truth": "ok",
            "experiment": "t",
            "condition": "clean",
            "question_id": "Q000002",
            "question_category": "No-Conflict",
        },
    ]
    stats = run_requests(Dummy(), jobs, out, dry_run=False, force=False, concurrency=1)
    assert stats["already_done"] == 1
    assert stats["to_run"] == 1
    assert calls["n"] == 1
