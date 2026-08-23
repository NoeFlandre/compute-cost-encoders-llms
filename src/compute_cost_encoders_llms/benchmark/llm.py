from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeGuard
from urllib.request import Request, urlopen

from ._numerics import logsumexp
from .example import candidate_labels, llm_prompt


class LlamaResponseError(ValueError):
    """Raised when llama.cpp does not return all required candidate scores."""


def completion_request_payload(prompt: str, seed: int) -> dict[str, object]:
    """Build a one-token, uncached llama.cpp logprob request."""

    return {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "seed": seed,
        "cache_prompt": False,
        "n_probs": 32,
        "timings_per_token": True,
        "stream": False,
    }


def chat_template_request_payload(user_prompt: str) -> dict[str, object]:
    """Build a Qwen-compatible user-message template request without thinking."""

    return {
        "messages": [{"role": "user", "content": user_prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


@dataclass(frozen=True, slots=True)
class LlamaScore:
    """One llama.cpp next-token logprob result and timing components."""

    logprobs: dict[str, float]
    tokenization_ms: float | None
    model_ms: float | None
    logprob_ms: float
    text_to_logprob_ms: float
    input_tokens: int | None


def _post_json(
    url: str, payload: Mapping[str, object], timeout_s: float
) -> Mapping[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout_s) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, Mapping):
        raise LlamaResponseError("llama.cpp response is not an object")
    return result


class LlamaClient:
    """Small HTTP client for one templated, uncached llama.cpp completion."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 300.0,
        request: Callable[
            [str, Mapping[str, object], float], Mapping[str, object]
        ] = _post_json,
    ) -> None:
        base = base_url.rstrip("/")
        self._template_url = base + "/apply-template"
        self._completion_url = base + "/completion"
        self._timeout_s = timeout_s
        self._request = request

    def score(self, seed: int) -> LlamaScore:
        """Request one next-token distribution for the fixed land-use prompt."""

        total_start = time.perf_counter_ns()
        template_response = self._request(
            self._template_url,
            chat_template_request_payload(llm_prompt()),
            self._timeout_s,
        )
        prompt = _rendered_prompt(template_response)
        response = self._request(
            self._completion_url,
            completion_request_payload(prompt, seed),
            self._timeout_s,
        )
        model_ms = _timing_ms(response)
        logprob_start = time.perf_counter_ns()
        logprobs = parse_candidate_logprobs(response, candidate_labels())
        logprob_ms = (time.perf_counter_ns() - logprob_start) / 1_000_000
        text_to_logprob_ms = (time.perf_counter_ns() - total_start) / 1_000_000
        return LlamaScore(
            logprobs=logprobs,
            tokenization_ms=None,
            model_ms=model_ms,
            logprob_ms=logprob_ms,
            text_to_logprob_ms=text_to_logprob_ms,
            input_tokens=_input_tokens(response),
        )


def _rendered_prompt(response: Mapping[str, object]) -> str:
    prompt = response.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise LlamaResponseError("llama.cpp template response has no rendered prompt")
    return prompt


def _timing_ms(response: Mapping[str, object]) -> float | None:
    timings = response.get("timings")
    if not isinstance(timings, Mapping):
        return None
    prompt_ms = _timing_value(timings, "prompt_ms")
    predicted_ms = _timing_value(timings, "predicted_ms")
    if prompt_ms is None:
        return None
    if predicted_ms is None:
        return None
    return _timing_total(prompt_ms, predicted_ms)


def _timing_value(timings: Mapping[str, object], field: str) -> float | None:
    value = timings.get(field)
    if value is None:
        return None
    if not _is_numeric_timing(value):
        raise LlamaResponseError("llama.cpp timings are not numeric")
    numeric = float(value)
    if not _is_valid_timing(numeric):
        raise LlamaResponseError("llama.cpp timings are invalid")
    return numeric


def _is_numeric_timing(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_valid_timing(value: float) -> bool:
    return math.isfinite(value) and value >= 0


def _timing_total(prompt_ms: float, predicted_ms: float) -> float:
    total = prompt_ms + predicted_ms
    if not _is_valid_timing(total):
        raise LlamaResponseError("llama.cpp timings are invalid")
    return total


def _input_tokens(response: Mapping[str, object]) -> int | None:
    sources = (
        (response.get("usage"), "prompt_tokens"),
        (response, "tokens_evaluated"),
        (response.get("timings"), "prompt_n"),
    )
    for source, field in sources:
        value = _mapping_entry(source, field)
        if value is not None:
            return _validated_token_count(value)
    return None


def _mapping_entry(source: object, field: str) -> object | None:
    if not isinstance(source, Mapping):
        return None
    return source.get(field)


def _validated_token_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LlamaResponseError("llama.cpp prompt token count is invalid")
    return value


def _entries(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    for loader in (_completion_entries, _choice_entries):
        entries = loader(payload)
        if entries is not None:
            return entries
    raise LlamaResponseError("response contains no completion logprobs")


def _completion_entries(
    payload: Mapping[str, object],
) -> list[Mapping[str, object]] | None:
    first = _first_list_item(payload.get("completion_probabilities"))
    probabilities = _list_field(first, "probs")
    if probabilities is not None:
        return _with_top_logprobs(probabilities)
    if _list_field(first, "top_logprobs") is not None:
        return _with_top_logprobs([first])
    return None


def _choice_entries(
    payload: Mapping[str, object],
) -> list[Mapping[str, object]] | None:
    first_choice = _first_list_item(payload.get("choices"))
    if not isinstance(first_choice, Mapping):
        return None
    content = _list_field(first_choice.get("logprobs"), "content")
    if content is None:
        return None
    return _with_top_logprobs(content)


def _first_list_item(value: object) -> object | None:
    if not isinstance(value, list) or not value:
        return None
    return value[0]


def _list_field(value: object, field: str) -> list[object] | None:
    if not isinstance(value, Mapping):
        return None
    entries = value.get(field)
    if not isinstance(entries, list):
        return None
    return entries


def _with_top_logprobs(raw_entries: object) -> list[Mapping[str, object]]:
    first = _first_logprob_entry(raw_entries)
    return [first, *_additional_logprob_entries(first)]


def _first_logprob_entry(raw_entries: object) -> Mapping[str, object]:
    if not isinstance(raw_entries, list) or not raw_entries:
        raise LlamaResponseError("response contains no token probabilities")
    first = raw_entries[0]
    if not isinstance(first, Mapping):
        raise LlamaResponseError("token probability entry is not an object")
    return first


def _additional_logprob_entries(
    first: Mapping[str, object],
) -> list[Mapping[str, object]]:
    top = first.get("top_logprobs")
    if not isinstance(top, list):
        return []
    return [item for item in top if isinstance(item, Mapping)]


def parse_candidate_logprobs(
    payload: Mapping[str, object], candidates: Sequence[str]
) -> dict[str, float]:
    """Extract raw log probabilities for all requested candidate labels."""

    candidate_by_key = {candidate.casefold(): candidate for candidate in candidates}
    scores: dict[str, list[float]] = {}
    for entry in _entries(payload):
        parsed = _candidate_logprob(entry, candidate_by_key)
        if parsed is not None:
            label, value = parsed
            scores.setdefault(label, []).append(value)
    _require_candidate_scores(scores, candidates)
    return {candidate: _logsumexp(scores[candidate]) for candidate in candidates}


def _candidate_logprob(
    entry: Mapping[str, object], candidate_by_key: Mapping[str, str]
) -> tuple[str, float] | None:
    parsed = _candidate_token_score(entry)
    if parsed is None:
        return None
    token, score = parsed
    normalized = token.strip().casefold()
    if normalized not in candidate_by_key:
        return None
    return candidate_by_key[normalized], score


def _candidate_token_score(entry: Mapping[str, object]) -> tuple[str, float] | None:
    token = entry.get("token")
    value = entry.get("logprob")
    if not isinstance(token, str):
        return None
    if not isinstance(value, (int, float)):
        return None
    score = float(value)
    if not math.isfinite(score):
        return None
    return token, score


def _require_candidate_scores(
    scores: Mapping[str, Sequence[float]], candidates: Sequence[str]
) -> None:
    missing = _missing_candidates(scores, candidates)
    if missing:
        raise LlamaResponseError(
            "response is missing candidate logprobs: " + ", ".join(missing)
        )


def _missing_candidates(
    scores: Mapping[str, Sequence[float]], candidates: Sequence[str]
) -> list[str]:
    return [candidate for candidate in candidates if candidate not in scores]


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise LlamaResponseError("candidate logprobs must not be empty")
    return logsumexp(values)
