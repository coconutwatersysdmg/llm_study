#!/usr/bin/env python
"""Send one tiny request per provider to verify API key + model snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clients.bailian import make_bailian_client
from clients.zhipu import make_zhipu_client
from config import MODELS, get_api_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["zhipu", "bailian"], required=True)
    args = parser.parse_args()

    key = get_api_key(args.provider)
    print(f"API key detected: {'YES' if bool(key) else 'NO'}")
    if not key:
        env = MODELS[args.provider]["api_key_env"]
        print(f"PROVIDER_UNAVAILABLE missing env={env}")
        sys.exit(1)

    meta = MODELS[args.provider]
    print(f"provider={args.provider}")
    print(f"model={meta['model']}")
    print(f"base_url={meta['base_url']}")
    if args.provider == "bailian":
        print(f"enable_thinking={meta.get('enable_thinking')}")

    client = make_zhipu_client() if args.provider == "zhipu" else make_bailian_client()
    resp = client.generate(
        system_prompt="只输出一个词。",
        user_prompt="回复：OK",
        request_id=f"check_api__{args.provider}",
    )
    returned = None
    if isinstance(resp.raw, dict):
        returned = resp.raw.get("model")
    print(
        {
            "success": resp.success,
            "returned_model_id": returned,
            "latency_sec": resp.latency_sec,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "total_tokens": resp.total_tokens,
            "attempts": resp.attempts,
            "error": resp.error,
            "raw_text_preview": (resp.raw_text or "")[:80],
        }
    )
    if not resp.success:
        print("PROVIDER_UNAVAILABLE")
        sys.exit(2)


if __name__ == "__main__":
    main()
