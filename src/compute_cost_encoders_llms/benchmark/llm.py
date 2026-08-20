from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from urllib.request import Request, urlopen

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


@dataclass(frozen=True, slots=True)
class LlamaScore:
    """One llama.cpp next-token logprob result and timing components."""

    logprobs: dict[str, float]
    tokenization_ms: float
    model_ms: float
    logprob_ms: float
    text_to_logprob_ms: float
    input_tokens: int


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
    """Small HTTP client for one uncached llama.cpp completion."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 300.0,
        request: Callable[
            [str, Mapping[str, object], float], Mapping[str, object]
        ] = _post_json,
    ) -> None:
        self._url = base_url.rstrip("/") + "/completion"
        self._timeout_s = timeout_s
        self._request = request

    def score(self, seed: int) -> LlamaScore:
        """Request one next-token distribution for the fixed land-use prompt."""

        total_start = time.perf_counter_ns()
        payload = completion_request_payload(llm_prompt(), seed)
        response = self._request(self._url, payload, self._timeout_s)
        model_ms = _timing_ms(response)
        tokenization_ms = 0.0
        logprob_start = time.perf_counter_ns()
        logprobs = parse_candidate_logprobs(response, candidate_labels())
        logprob_ms = (time.perf_counter_ns() - logprob_start) / 1_000_000
        text_to_logprob_ms = (time.perf_counter_ns() - total_start) / 1_000_000
        return LlamaScore(
            logprobs=logprobs,
            tokenization_ms=tokenization_ms,
            model_ms=model_ms,
            logprob_ms=logprob_ms,
            text_to_logprob_ms=text_to_logprob_ms,
            input_tokens=_input_tokens(response),
        )


def _timing_ms(response: Mapping[str, object]) -> float:
    timings = response.get("timings")
    if not isinstance(timings, Mapping):
        return 0.0
    prompt_ms = timings.get("prompt_ms", 0.0)
    predicted_ms = timings.get("predicted_ms", 0.0)
    if not isinstance(prompt_ms, (int, float)) or not isinstance(
        predicted_ms, (int, float)
    ):
        raise LlamaResponseError("llama.cpp timings are not numeric")
    return float(prompt_ms) + float(predicted_ms)


def _input_tokens(response: Mapping[str, object]) -> int:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return 0
    prompt_tokens = usage.get("prompt_tokens", 0)
    if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        raise LlamaResponseError("llama.cpp prompt token count is invalid")
    return prompt_tokens


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

    scores: dict[str, float] = {}
    for entry in _entries(payload):
        parsed = _candidate_logprob(entry, candidates)
        if parsed is not None:
            label, value = parsed
            scores[label] = value
    _require_candidate_scores(scores, candidates)
    return {candidate: scores[candidate] for candidate in candidates}


def _candidate_logprob(
    entry: Mapping[str, object], candidates: Sequence[str]
) -> tuple[str, float] | None:
    token = entry.get("token")
    value = entry.get("logprob")
    if not isinstance(token, str) or not isinstance(value, (int, float)):
        return None
    normalized = token.strip().lower()
    if normalized not in candidates or not math.isfinite(float(value)):
        return None
    return normalized, float(value)


def _require_candidate_scores(
    scores: Mapping[str, float], candidates: Sequence[str]
) -> None:
    missing = _missing_candidates(scores, candidates)
    if missing:
        raise LlamaResponseError(
            "response is missing candidate logprobs: " + ", ".join(missing)
        )


def _missing_candidates(
    scores: Mapping[str, float], candidates: Sequence[str]
) -> list[str]:
    return [candidate for candidate in candidates if candidate not in scores]
