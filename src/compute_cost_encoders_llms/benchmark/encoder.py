from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, TypeGuard, runtime_checkable

from ._numerics import logsumexp
from .example import candidate_label_forms, candidate_labels, encoder_prompt


@runtime_checkable
class TensorLike(Protocol):
    def __getitem__(self, index: int | slice) -> TensorLike: ...

    def to(self, device: str) -> TensorLike: ...

    def tolist(self) -> object: ...

    def detach(self) -> TensorLike: ...

    def float(self) -> TensorLike: ...

    def cpu(self) -> TensorLike: ...


@runtime_checkable
class TokenizerLike(Protocol):
    mask_token: str
    mask_token_id: int

    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]: ...


@runtime_checkable
class ModelLike(Protocol):
    def __call__(self, **inputs: object) -> ModelOutputLike: ...


class ModelOutputLike(Protocol):
    logits: object


@runtime_checkable
class TorchLike(Protocol):
    def inference_mode(self) -> AbstractContextManager[object]: ...


def validate_single_token_candidates(
    candidate_tokens: Mapping[str, Sequence[int]],
) -> dict[str, int]:
    """Return candidate token IDs, rejecting labels split into multiple tokens."""

    result: dict[str, int] = {}
    for label in candidate_labels():
        tokens = candidate_tokens.get(label)
        if tokens is None or len(tokens) != 1:
            raise ValueError(f"{label} must be represented by a single token")
        result[label] = int(tokens[0])
    return result


def validate_single_token_variants(
    candidate_tokens: Mapping[str, Sequence[Sequence[int]]],
) -> dict[str, tuple[int, ...]]:
    """Return exact vocabulary IDs for all single-token label spellings."""

    result: dict[str, tuple[int, ...]] = {}
    for label in candidate_labels():
        forms = candidate_tokens.get(label)
        result[label] = _validated_variant_ids(label, forms)
    return result


def _validated_variant_ids(
    label: str, forms: Sequence[Sequence[int]] | None
) -> tuple[int, ...]:
    if not forms:
        raise ValueError(f"{label} must have a single-token form")
    token_ids = tuple(_validated_single_token_id(label, tokens) for tokens in forms)
    return tuple(dict.fromkeys(token_ids))


def _validated_single_token_id(label: str, tokens: Sequence[int]) -> int:
    if len(tokens) != 1:
        raise ValueError(f"{label} must be represented by a single token")
    return int(tokens[0])


def mask_position(input_ids: Sequence[int], mask_token_id: int) -> int:
    """Return the only masked position in an encoded input."""

    positions = [
        index for index, token_id in enumerate(input_ids) if token_id == mask_token_id
    ]
    if len(positions) != 1:
        raise ValueError("input must contain exactly one mask token")
    return positions[0]


def candidate_logprobs(
    logits: Sequence[float], candidate_token_ids: Mapping[str, int]
) -> dict[str, float]:
    """Return log-softmax scores for selected candidate token IDs."""

    values = _validated_logits(logits)
    normalizer = _log_normalizer(values)
    return {
        label: _candidate_logprob(values, token_id, normalizer)
        for label, token_id in candidate_token_ids.items()
    }


def candidate_variant_logprobs(
    logits: Sequence[float], candidate_token_ids: Mapping[str, Sequence[int]]
) -> dict[str, float]:
    """Aggregate log-softmax scores for exact token-ID label variants."""

    values = _validated_logits(logits)
    normalizer = _log_normalizer(values)
    return {
        label: _variant_logprob(values, token_ids, normalizer)
        for label, token_ids in candidate_token_ids.items()
    }


def _validated_logits(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("logits must not be empty")
    values = [float(value) for value in logits]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("logits must be finite")
    return values


def _log_normalizer(logits: Sequence[float]) -> float:
    return logsumexp(logits)


def _candidate_logprob(
    logits: Sequence[float], token_id: int, normalizer: float
) -> float:
    if token_id < 0 or token_id >= len(logits):
        raise ValueError(f"candidate token ID is out of range: {token_id}")
    return float(logits[token_id]) - normalizer


def _variant_logprob(
    logits: Sequence[float], token_ids: Sequence[int], normalizer: float
) -> float:
    if not token_ids:
        raise ValueError("candidate token IDs must not be empty")
    values = [
        _candidate_logprob(logits, token_id, normalizer) for token_id in token_ids
    ]
    return logsumexp(values)


def _tensor_like(value: object) -> TensorLike:
    if not isinstance(value, TensorLike):
        raise ValueError("encoder tensor value is invalid")
    return value


def _is_integer_list(value: object) -> TypeGuard[list[int]]:
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def _integer_list(value: object) -> list[int]:
    if not _is_integer_list(value):
        raise ValueError("encoder token IDs are invalid")
    return value


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("encoder logits are invalid")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("encoder logits are invalid")
        result.append(float(item))
    return result


@dataclass(frozen=True, slots=True)
class EncoderScore:
    """One masked-language-model score and its timing components."""

    logprobs: dict[str, float]
    tokenization_ms: float
    model_ms: float
    logprob_ms: float
    text_to_logprob_ms: float
    input_tokens: int


def score_transformers_once(
    tokenizer: TokenizerLike,
    model: ModelLike,
    torch_module: TorchLike,
    device: str,
) -> EncoderScore:
    """Score yes/no at one masked position using a Transformers model."""

    total_start = time.perf_counter_ns()
    token_start = total_start
    encoded = tokenizer(
        encoder_prompt(tokenizer.mask_token),
        return_tensors="pt",
        add_special_tokens=True,
    )
    candidate_tokens = {
        label: tuple(
            _integer_list(tokenizer(form, add_special_tokens=False)["input_ids"])
            for form in candidate_label_forms(label)
        )
        for label in candidate_labels()
    }
    tokenization_ms = (time.perf_counter_ns() - token_start) / 1_000_000
    input_ids = _tensor_like(encoded["input_ids"])
    input_id_list = _integer_list(input_ids[0].tolist())
    position = mask_position(input_id_list, tokenizer.mask_token_id)
    candidate_ids = validate_single_token_variants(candidate_tokens)
    model_inputs = {
        name: _tensor_like(value).to(device) for name, value in encoded.items()
    }

    _synchronize(torch_module)
    model_start = time.perf_counter_ns()
    with torch_module.inference_mode():
        outputs = model(**model_inputs)
    _synchronize(torch_module)
    model_ms = (time.perf_counter_ns() - model_start) / 1_000_000

    logprob_start = time.perf_counter_ns()
    logits = _tensor_like(outputs.logits)[0][position]
    logits = _float_list(logits.detach().float().cpu().tolist())
    logprobs = candidate_variant_logprobs(logits, candidate_ids)
    logprob_ms = (time.perf_counter_ns() - logprob_start) / 1_000_000
    text_to_logprob_ms = (time.perf_counter_ns() - total_start) / 1_000_000
    return EncoderScore(
        logprobs=logprobs,
        tokenization_ms=tokenization_ms,
        model_ms=model_ms,
        logprob_ms=logprob_ms,
        text_to_logprob_ms=text_to_logprob_ms,
        input_tokens=len(input_id_list),
    )


def _synchronize(torch_module: object) -> None:
    cuda = getattr(torch_module, "cuda", None)
    synchronize = getattr(cuda, "synchronize", None)
    if callable(synchronize):
        synchronize()
