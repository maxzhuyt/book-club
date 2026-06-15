"""
Summarize a book-club run with Anthropic Claude Opus 4.6 via OpenRouter.

Supports both layouts:
- outputs/ : single-story discussion (story_1)
- outputs_compare/ : two-version comparative discussion (A vs B)

The script auto-detects which layout it is given (presence of
manifest.json key 'story_a_title' -> comparative) and dispatches to the
right prompt builder.  Saves to:
- outputs/group_summary_opus.md       (does NOT overwrite group_summary.md)
- outputs_compare/group_summary_opus.md

API keys come from ~/.openrouter_keys (named JSON entries).  Default key
name is "Hoyt"; override with --key-name.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests


# Reuse the existing input-builder + vote parser
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from bookclub.summarize_compare import (
    SUMMARIZER_SYSTEM as COMPARE_SYSTEM,
    parse_votes,
    build_user_msg as build_compare_user_msg,
)
from bookclub.summarize import (
    SUMMARIZER_SYSTEM as SINGLE_SYSTEM,
    build_user_msg as build_single_user_msg,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_URL   = "https://api.deepseek.com/v1/chat/completions"
MODEL_ID = "anthropic/claude-opus-4.6"  # OpenRouter id for Opus 4.6
DEEPSEEK_MODEL_ID = "deepseek-chat"


def load_key(key_name: str, backend: str = "openrouter") -> str:
    if backend == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise SystemExit("DEEPSEEK_API_KEY not set in environment")
        return key
    path = Path.home() / ".openrouter_keys"
    if not path.exists():
        raise SystemExit(f"~/.openrouter_keys not found")
    keys = json.loads(path.read_text())
    if key_name not in keys:
        raise SystemExit(
            f"Key '{key_name}' not found in ~/.openrouter_keys. "
            f"Available: {list(keys.keys())}"
        )
    return keys[key_name]


def detect_layout(outputs_dir: Path) -> str:
    manifest = json.loads((outputs_dir / "manifest.json").read_text())
    return "compare" if "story_a_title" in manifest else "single"


def call_opus(system: str, user: str, api_key: str,
              max_tokens: int = 2000, temperature: float = 0.3,
              model_id: str = MODEL_ID, backend: str = "openrouter") -> dict:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if backend == "deepseek":
        url = DEEPSEEK_URL
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    else:
        url = OPENROUTER_URL
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/contrastive-decoding-bookclub",
            "X-Title": "Book Club CD Simulation Summary",
        }
    print(f"[opus] calling {model_id} via {backend} (max_tokens={max_tokens})...",
          flush=True)
    t0 = time.time()
    resp = requests.post(url, json=payload, headers=headers,
                         timeout=600)
    dt = time.time() - t0
    if resp.status_code != 200:
        raise SystemExit(
            f"OpenRouter error {resp.status_code}: {resp.text[:1000]}"
        )
    data = resp.json()
    print(f"[opus] done in {dt:.1f}s "
          f"(prompt={data.get('usage', {}).get('prompt_tokens', '?')}, "
          f"completion={data.get('usage', {}).get('completion_tokens', '?')})",
          flush=True)
    return data


def summarize_single(outputs_dir: Path, api_key: str,
                     model_id: str, backend: str = "openrouter") -> tuple[str, str]:
    manifest = json.loads((outputs_dir / "manifest.json").read_text())
    title = manifest.get("story_title", "the story")
    user_msg = build_single_user_msg(
        outputs_dir / "transcripts", manifest, title,
    )
    input_path = outputs_dir / "summary_input.md"
    if not input_path.exists():
        input_path.write_text(user_msg)
    data = call_opus(SINGLE_SYSTEM, user_msg, api_key,
                     max_tokens=1800, model_id=model_id, backend=backend)
    text = data["choices"][0]["message"]["content"]
    return text, json.dumps(data.get("usage", {}), indent=2)


def summarize_compare(outputs_dir: Path, api_key: str,
                      model_id: str, backend: str = "openrouter") -> tuple[str, str]:
    manifest = json.loads((outputs_dir / "manifest.json").read_text())
    p4 = json.loads((outputs_dir / "transcripts"
                     / "phase4_reflections.json").read_text())
    votes = parse_votes(p4)
    (outputs_dir / "votes.json").write_text(json.dumps(votes, indent=2))
    user_msg = build_compare_user_msg(
        outputs_dir / "transcripts", manifest, votes,
    )
    input_path = outputs_dir / "summary_input.md"
    if not input_path.exists():
        input_path.write_text(user_msg)
    data = call_opus(COMPARE_SYSTEM, user_msg, api_key,
                     max_tokens=2200, model_id=model_id, backend=backend)
    text = data["choices"][0]["message"]["content"]
    return text, json.dumps(data.get("usage", {}), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", type=Path, required=True,
                    help="e.g. simulation/outputs_compare or simulation/outputs")
    ap.add_argument("--key-name", type=str, default="Hoyt",
                    help="Named key in ~/.openrouter_keys (openrouter backend only)")
    ap.add_argument("--model-id", type=str, default=None,
                    help="Model id (defaults to backend-appropriate model)")
    ap.add_argument("--backend", type=str, default="openrouter",
                    choices=["openrouter", "deepseek"],
                    help="API backend to use")
    ap.add_argument("--out-name", type=str, default="group_summary_opus.md")
    args = ap.parse_args()

    if args.model_id is None:
        args.model_id = DEEPSEEK_MODEL_ID if args.backend == "deepseek" else MODEL_ID

    api_key = load_key(args.key_name, backend=args.backend)
    layout = detect_layout(args.outputs_dir)
    print(f"[opus] layout={layout}, backend={args.backend}, "
          f"model={args.model_id}, outputs_dir={args.outputs_dir}",
          flush=True)

    if layout == "compare":
        text, usage = summarize_compare(args.outputs_dir, api_key,
                                        args.model_id, backend=args.backend)
    else:
        text, usage = summarize_single(args.outputs_dir, api_key,
                                       args.model_id, backend=args.backend)

    out_path = args.outputs_dir / args.out_name
    out_path.write_text(text)
    (args.outputs_dir / "opus_usage.json").write_text(usage)
    print(f"[opus] wrote {out_path}", flush=True)
    print("\n----- SUMMARY -----\n" + text, flush=True)


if __name__ == "__main__":
    main()
