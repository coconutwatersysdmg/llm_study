#!/usr/bin/env python
"""Validate frozen dataset hashes without regenerating data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils.io_utils import read_json, sha256_file


def main() -> int:
    q_path = config.QA_DIR / "questions.jsonl"
    e_path = config.QA_DIR / "evidence.jsonl"
    m_path = config.QA_DIR / "dataset_manifest.json"
    for p in (q_path, e_path, m_path):
        if not p.exists():
            print(f"STOP: missing {p}")
            return 2

    manifest = read_json(m_path)
    q_count = manifest.get("question_count")
    cats = manifest.get("category_counts", {})
    print(f"dataset_version={manifest.get('dataset_version')}")
    print(f"question_count={q_count}")
    print(f"category_counts={cats}")

    if q_count != 1200:
        print("STOP: question_count != 1200")
        return 2

    sha_q = sha256_file(q_path)
    sha_e = sha256_file(e_path)
    print(f"SHA256_questions={sha_q}")
    print(f"SHA256_evidence={sha_e}")
    print(f"manifest_SHA256_questions={manifest.get('SHA256_questions')}")
    print(f"manifest_SHA256_evidence={manifest.get('SHA256_evidence')}")

    if sha_q != manifest.get("SHA256_questions") or sha_e != manifest.get("SHA256_evidence"):
        print("STOP: DATASET_HASH_MISMATCH")
        return 3

    print("DATASET_HASH_OK")
    print(f"dataset_sha256_questions={sha_q}")
    print(f"dataset_sha256_evidence={sha_e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
