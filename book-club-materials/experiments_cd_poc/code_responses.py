"""Code CD-PoC reader responses with the v3 attention coder.

Same coder model and prompts as experiments_v3 (deepseek-v4-flash, temp 0.2)
so codings are comparable to the v3 run. v3 reached it via OpenRouter; the
OpenRouter key here is dead, so we use the DeepSeek direct API, which serves
the same `deepseek-v4-flash` model id. Key: DEEPSEEK_API_KEY from ~/.env.

Walks run_cd_poc/alpha*/results/**/agent-*/ and writes coding.json next to
pass1.txt/pass2.txt. Resumable: skips agents whose coding.json exists.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

V3_DIR = Path("/project/jevans/maxzhuyt/book-club-v3/book-club-materials/experiments_v3")
sys.path.insert(0, str(V3_DIR))
from coding import CODER_SYSTEM, DIMENSIONS, parse_coding  # noqa: E402

URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"


def load_key() -> str:
    for line in (Path.home() / ".env").read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("no DEEPSEEK_API_KEY found")
    return key


def code_text(text: str, key: str, retries: int = 3) -> dict:
    user = f"{DIMENSIONS}\n\n=== READER RESPONSE ===\n{text}\n=== END ==="
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": CODER_SYSTEM},
                     {"role": "user", "content": user}],
        "max_tokens": 400, "temperature": 0.2,
        "thinking": {"type": "disabled"},   # match v3's reasoning-excluded coder
    }
    last = None
    for i in range(retries):
        try:
            r = requests.post(URL, json=payload, timeout=120,
                              headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            obj = parse_coding(content)
            if "_parse_error" not in obj:
                return obj
            last = obj
        except Exception as e:
            last = {"_parse_error": str(e)}
        time.sleep(2 * (i + 1))
    return last or {"_parse_error": "exhausted"}


def main():
    base = Path(__file__).resolve().parent / "run_cd_poc"
    key = load_key()
    agents = sorted(base.glob("alpha*/results/*/*/*/agent-*"))
    print(f"{len(agents)} agent dirs found")
    done = skipped = 0
    for d in agents:
        out = d / "coding.json"
        if out.exists():
            skipped += 1
            continue
        p1, p2 = d / "pass1.txt", d / "pass2.txt"
        if not p2.exists():
            continue
        coding = {
            "pass1": code_text(p1.read_text(), key),
            "pass2": code_text(p2.read_text(), key),
        }
        out.write_text(json.dumps(coding, indent=1))
        done += 1
        if done % 20 == 0:
            print(f"coded {done} (skipped {skipped})", flush=True)
    print(f"DONE coded={done} skipped={skipped}")


if __name__ == "__main__":
    main()
