"""Cheap LLM judges over OpenRouter's chat API (benchmark only).

They get exactly LLMJudge's faithfulness rubric, user message, and JSON schema, so
DeepSeek and GLM grade with the same instructions as Claude; only the transport and
the model differ. A failed or unparseable call is recorded as an error row with its
cost — an invalid answer is a real failure mode of LLM judges and gets counted.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from ragverdict.bench.predict import Prediction, PredictionStore, collect, pending
from ragverdict.bench.ragtruth import Example
from ragverdict.judges.base import JudgeError, JudgeScore, JudgeTransportError
from ragverdict.judges.llm_judge import faithfulness_prompt, output_schema

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class ChatJudgeConfig:
    model: str
    max_tokens: int
    reasoning: dict[str, Any] | None = None  # OpenRouter `reasoning` param; None = omit
    temperature: float | None = 0.0  # None = omit (provider default)
    json_mode: Literal["json_schema", "json_object"] = "json_schema"


# Mirrors the 2026-09-19 pilot: DeepSeek thinking off / temp 0 / 512 tokens;
# GLM thinking on (effort low) / sampling default / 4096 tokens.
DEEPSEEK_FLASH = ChatJudgeConfig(
    "deepseek/deepseek-v4.1-flash", 512, reasoning={"enabled": False}, temperature=0.0
)
GLM_FLASH = ChatJudgeConfig("z-ai/glm-5.3-flash", 4096, reasoning={"effort": "low"}, temperature=None)


def chat_body(cfg: ChatJudgeConfig, response_text: str, source: str) -> dict[str, Any]:
    system, user = faithfulness_prompt(response_text, source)
    schema = output_schema(JudgeScore)
    if cfg.json_mode == "json_schema":
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {"name": "JudgeScore", "strict": True, "schema": schema},
        }
    else:  # model lacks structured outputs: plain JSON mode, schema stated in the prompt
        response_format = {"type": "json_object"}
        system = f"{system}\n\nJSON schema for your output:\n{json.dumps(schema)}"
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": response_format,
        "max_tokens": cfg.max_tokens,
        "usage": {"include": True},
        "provider": {"require_parameters": True},
    }
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature
    if cfg.reasoning is not None:
        body["reasoning"] = cfg.reasoning
    return body


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def parse_chat_response(data: dict[str, Any]) -> JudgeScore:
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeError(f"unexpected response shape: {str(data)[:200]}") from exc
    if choice.get("finish_reason") == "length":
        raise JudgeError("judge output truncated (finish_reason=length)")
    try:
        return JudgeScore.model_validate_json(_strip_fences(content))
    except ValidationError as exc:
        raise JudgeError(f"judge returned invalid JSON for JudgeScore: {str(exc)[:200]}") from exc


def _post(
    client: httpx.Client, body: dict[str, Any], api_key: str, max_attempts: int, backoff_s: float
) -> dict[str, Any]:
    last = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.post(
                OPENROUTER_CHAT_URL, json=body, headers={"Authorization": f"Bearer {api_key}"}
            )
        except httpx.HTTPError as exc:
            last = f"network error: {exc}"
        else:
            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise JudgeError("non-JSON body from OpenRouter") from exc
                if not isinstance(data, dict):
                    raise JudgeError(f"unexpected response shape: {str(data)[:200]}")
                return data
            last = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code not in _RETRYABLE_STATUS:
                raise JudgeTransportError(last)
        if attempt < max_attempts:
            time.sleep(backoff_s * 2 ** (attempt - 1))
    raise JudgeTransportError(f"failed after {max_attempts} attempts: {last}")


def run_chat_judge(
    examples: Sequence[Example],
    cfg: ChatJudgeConfig,
    store: PredictionStore,
    run: str,
    *,
    api_key: str,
    client: httpx.Client | None = None,
    workers: int = 8,
    repeats: int = 1,
    max_attempts: int = 3,
    backoff_s: float = 1.0,
) -> list[Prediction]:
    http = client or httpx.Client(timeout=120.0)

    def one(item: tuple[Example, int]) -> None:
        ex, repeat = item
        started = time.perf_counter()
        data: dict[str, Any] = {}
        error_kind: str | None = None
        try:
            data = _post(http, chat_body(cfg, ex.response, ex.source), api_key, max_attempts, backoff_s)
            score: JudgeScore | None = parse_chat_response(data)
            error = None
        except JudgeError as exc:
            score, error = None, str(exc)
            error_kind = "transport" if isinstance(exc, JudgeTransportError) else "judge"
        usage_raw = data.get("usage")
        usage: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
        served_model = str(data.get("model", cfg.model))
        provider = data.get("provider")
        if provider:
            served_model = f"{served_model} via {provider}"
        store.append(
            Prediction(
                run=run,
                example_id=ex.id,
                repeat=repeat,
                score=score.score if score else None,
                input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage.get("completion_tokens", 0) or 0),
                cost_usd=float(usage.get("cost", 0.0) or 0.0),
                latency_s=time.perf_counter() - started,
                served_model=served_model,
                reasoning=score.reasoning if score else "",
                error=error,
                error_kind=error_kind,
            )
        )

    todo = pending(examples, store.load(run), repeats)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    return collect(store, run, examples, repeats)
