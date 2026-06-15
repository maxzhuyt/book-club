"""
Shared OpenRouter client for the closed-model plans.

Mirrors the pattern in simulation/src/bookclub/summarize_opus.py:
- Keys come from ~/.openrouter_keys (JSON, named entries).
- Default key name is "Hoyt".
- Generic chat-completion call returns the full response dict.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_KEY_NAME = "Hoyt"
DEFAULT_REFERER = "https://localhost/contrastive-decoding-bookclub"
DEFAULT_TITLE = "Book Club CD-on-Closed-Models"


def load_key(key_name: str = DEFAULT_KEY_NAME) -> str:
    path = Path.home() / ".openrouter_keys"
    if not path.exists():
        raise SystemExit(f"~/.openrouter_keys not found at {path}")
    keys = json.loads(path.read_text())
    if key_name not in keys:
        raise SystemExit(
            f"Key '{key_name}' not in ~/.openrouter_keys. "
            f"Available: {list(keys.keys())}"
        )
    return keys[key_name]


def call_openrouter(
    *,
    model_id: str,
    system: str,
    user: str,
    api_key: str,
    max_tokens: int = 1200,
    temperature: float = 0.75,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    timeout: int = 600,
    referer: str = DEFAULT_REFERER,
    title: str = DEFAULT_TITLE,
    extra_headers: Optional[dict] = None,
    retries: int = 3,
    retry_backoff: float = 4.0,
) -> dict:
    """One chat-completion call. Returns the parsed JSON response."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        payload["seed"] = int(seed)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": title,
    }
    if extra_headers:
        headers.update(extra_headers)

    last_err: Optional[str] = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                OPENROUTER_URL, json=payload, headers=headers, timeout=timeout,
            )
        except requests.RequestException as exc:
            last_err = f"network: {exc}"
            time.sleep(retry_backoff * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.json()
        # 429 / 5xx are worth retrying
        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = f"{resp.status_code}: {resp.text[:300]}"
            time.sleep(retry_backoff * (attempt + 1))
            continue
        raise SystemExit(
            f"OpenRouter error {resp.status_code}: {resp.text[:1000]}"
        )
    raise SystemExit(f"OpenRouter failed after {retries} retries: {last_err}")


def extract_text(resp: dict) -> str:
    return resp["choices"][0]["message"]["content"]


def call_openrouter_n_parallel(
    *,
    model_id: str,
    system: str,
    user: str,
    api_key: str,
    n: int,
    base_seed: int,
    max_tokens: int = 1200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_workers: int = 6,
    timeout: int = 600,
) -> list[dict]:
    """
    Fire N concurrent requests with different seeds. Returns N responses
    in seed order (not completion order).
    """
    seeds = [base_seed + i for i in range(n)]

    def one(seed: int) -> tuple[int, dict]:
        resp = call_openrouter(
            model_id=model_id, system=system, user=user, api_key=api_key,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            seed=seed, timeout=timeout,
        )
        return seed, resp

    out: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
        futs = [pool.submit(one, s) for s in seeds]
        for f in as_completed(futs):
            seed, resp = f.result()
            out[seed] = resp
    return [out[s] for s in seeds]
