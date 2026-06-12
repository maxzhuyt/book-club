"""OpenRouter async client for deepseek-v4-flash. Copied from V1 (run_experiment.py)
with its resilience: hard per-call timeout, retry with token escalation on empty/length,
reasoning channel disabled, reasoning-field fallback."""
from __future__ import annotations
import asyncio, os
from pathlib import Path

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-v4-flash"
API_KEY_ENV_VAR = "NARRATIVE"
PER_CALL_TIMEOUT = 180.0
_client = None


def _dotenv_candidates():
    """Locations to search for a .env file, in priority order:
       1. $DOTENV_PATH if set
       2. ~/.env
       3. <this-file-dir>/.env and each ancestor's .env walking up the tree
    """
    if os.environ.get("DOTENV_PATH"):
        yield Path(os.environ["DOTENV_PATH"])
    yield Path.home() / ".env"
    here = Path(__file__).resolve().parent
    for ancestor in (here, *here.parents):
        yield ancestor / ".env"


def load_dotenv_into_environ():
    for p in _dotenv_candidates():
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return  # first found wins


def get_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI
        load_dotenv_into_environ()
        key = os.environ.get(API_KEY_ENV_VAR, "")
        if not key:
            raise RuntimeError(
                f"{API_KEY_ENV_VAR} not set. Either export it, place it in a "
                f"~/.env file, or set DOTENV_PATH to a .env file containing it."
            )
        _client = AsyncOpenAI(api_key=key, base_url=OPENROUTER_BASE_URL,
                              timeout=PER_CALL_TIMEOUT, max_retries=0)
    return _client


async def chat(messages: list[dict], max_tokens: int, temperature: float,
               max_retries: int = 2) -> str:
    """messages: full OpenAI-style list. Returns assistant text or raises."""
    client = get_client()
    last_err, cur = None, max_tokens
    for attempt in range(max_retries + 1):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL, messages=messages, max_tokens=cur,
                    temperature=temperature, extra_body={"reasoning": {"exclude": True}},
                ),
                timeout=PER_CALL_TIMEOUT + 30,
            )
            msg = resp.choices[0].message
            finish = resp.choices[0].finish_reason
            if msg.content and msg.content.strip():
                return msg.content.strip()
            dump = msg.model_dump()
            r = dump.get("reasoning") or dump.get("reasoning_content")
            if r and isinstance(r, str) and r.strip():
                return r.strip()
            last_err = f"empty content (finish={finish})"
            if finish == "length":
                cur = min(int(cur * 1.8), 16000)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < max_retries:
            await asyncio.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"chat() failed after {max_retries + 1} attempts: {last_err}")
