from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from types import SimpleNamespace
from typing import ClassVar
from urllib.request import Request

import pytest

import compute_cost_encoders_llms.benchmark._numerics as numerics_module
import compute_cost_encoders_llms.benchmark.encoder as encoder_module
import compute_cost_encoders_llms.benchmark.llm as llm_module
from compute_cost_encoders_llms.benchmark._numerics import logsumexp
from compute_cost_encoders_llms.benchmark.config import BenchmarkConfig, ConfigError
from compute_cost_encoders_llms.benchmark.encoder import (
    _float_list,
    _integer_list,
    _tensor_like,
    _variant_logprob,
    candidate_logprobs,
    candidate_variant_logprobs,
    mask_position,
    score_transformers_once,
    validate_single_token_candidates,
    validate_single_token_variants,
)
from compute_cost_encoders_llms.benchmark.example import (
    LANDUSE_SENTENCE,
    candidate_label_forms,
    candidate_labels,
    encoder_prompt,
    llm_prompt,
)
from compute_cost_encoders_llms.benchmark.llm import (
    LlamaClient,
    LlamaResponseError,
    _entries,
    _first_logprob_entry,
    _input_tokens,
    _logsumexp,
    _post_json,
    _require_candidate_scores,
    _timing_ms,
    _timing_total,
    _timing_value,
    _validated_token_count,
    _with_top_logprobs,
    chat_template_request_payload,
    completion_request_payload,
    parse_candidate_logprobs,
)


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.to_devices: list[str] = []

    def __getitem__(self, index):
        return FakeTensor(self.value[index])

    def to(self, _device):
        self.to_devices.append(_device)
        return self

    def tolist(self):
        return self.value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self


class FakeTokenizer:
    mask_token = "<mask>"
    mask_token_id = 99

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.last_encoded: dict[str, FakeTensor] | None = None

    def __call__(self, text, **_kwargs):
        self.calls.append((text, dict(_kwargs)))
        candidate_ids = {
            "yes": [1],
            "Yes": [3],
            "YES": [5],
            "no": [2],
            "No": [4],
            "NO": [6],
        }
        if text in candidate_ids:
            return {"input_ids": candidate_ids[text]}
        encoded = {
            "input_ids": FakeTensor([[10, 99, 11]]),
            "attention_mask": FakeTensor([[1, 1, 1]]),
        }
        self.last_encoded = encoded
        return encoded


class FakeTorch:
    class _InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    @staticmethod
    def inference_mode():
        return FakeTorch._InferenceMode()


class FakeModel:
    def __call__(self, **_inputs):
        return SimpleNamespace(
            logits=FakeTensor(
                [
                    [
                        [0.0] * 7,
                        [0.0, 2.0, -1.0, 1.5, -2.0, 1.0, -3.0],
                        [0.0] * 7,
                    ]
                ]
            )
        )


class JsonResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b'{"ok": true}'


class EncodedText(str):
    encodings: ClassVar[list[str]] = []

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        self.encodings.append(encoding)
        return super().encode(encoding, errors)


class EncodedBytes(bytes):
    decodings: ClassVar[list[str]] = []

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        self.decodings.append(encoding)
        return super().decode(encoding, errors)


class EncodedJsonResponse(JsonResponse):
    def read(self) -> EncodedBytes:
        return EncodedBytes(b'{"ok": true}')


class CapturedRequest:
    def __init__(self, url: str, *, data: bytes, headers: dict[str, str]) -> None:
        self.values: dict[str, object] = {
            "url": url,
            "data": data,
            "headers": headers,
        }


def test_landuse_example_is_binary_and_prompts_share_the_sentence() -> None:
    labels = candidate_labels()

    assert LANDUSE_SENTENCE == (
        "A public park with grass, trees, and walking paths occupies the parcel."
    )
    assert labels == ("yes", "no")
    assert LANDUSE_SENTENCE in encoder_prompt("<mask>")
    assert LANDUSE_SENTENCE in llm_prompt()


def test_prompts_ask_relevance_of_the_target_sentence() -> None:
    assert encoder_prompt("<mask>") == (
        'Here is a target sentence: "'
        + LANDUSE_SENTENCE
        + '"\nIs this sentence relevant for a land use description?\nAnswer: <mask>'
    )
    assert llm_prompt() == (
        'Here is a target sentence: "'
        + LANDUSE_SENTENCE
        + '"\nIs this sentence relevant for a land use description? '
        "Answer with exactly yes or no."
    )


def test_candidate_label_forms_cover_case_variants() -> None:
    assert candidate_label_forms("yes") == ("yes", "Yes", "YES")
    assert candidate_label_forms("no") == ("no", "No", "NO")
    with pytest.raises(
        ValueError,
        match=r"^unsupported candidate label: maybe$",
    ):
        candidate_label_forms("maybe")


def test_prompts_reject_invalid_mask_tokens_with_exact_error() -> None:
    with pytest.raises(ValueError, match=r"^mask_token must not be empty$"):
        encoder_prompt("")


def test_configuration_requires_immutable_revisions_and_positive_repetitions() -> None:
    config = BenchmarkConfig(
        encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
        llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
        llama_cpp_revision="test-llama-cpp-revision",
        repetitions=4,
        warmups=2,
    )

    assert config.repetitions == 4

    boundary = BenchmarkConfig(
        encoder_revision=config.encoder_revision,
        llm_revision=config.llm_revision,
        llama_cpp_revision=config.llama_cpp_revision,
        repetitions=1,
        warmups=0,
    )
    assert boundary.repetitions == 1
    assert boundary.warmups == 0

    with pytest.raises(ConfigError, match="encoder_revision"):
        BenchmarkConfig(
            encoder_revision="main",
            llm_revision=config.llm_revision,
            llama_cpp_revision=config.llama_cpp_revision,
        )


def test_configuration_run_settings_have_exact_validation_errors() -> None:
    with pytest.raises(ConfigError, match=r"^repetitions must be positive$"):
        BenchmarkConfig(
            encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
            llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
            llama_cpp_revision="test-llama-cpp-revision",
            repetitions=0,
        )
    with pytest.raises(ConfigError, match=r"^warmups must be non-negative$"):
        BenchmarkConfig(
            encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
            llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
            llama_cpp_revision="test-llama-cpp-revision",
            warmups=-1,
        )


def test_configuration_loads_benchmark_toml(tmp_path) -> None:
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(
        """
[benchmark]
encoder_revision = "c5955035435e2bf121cde7f3c8863ef52ff35d82"
llm_revision = "8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8"
llama_cpp_revision = "6503355df0eb4f65875012523263c302fe0088c1"
repetitions = 3
warmups = 1
""".strip()
    )

    config = BenchmarkConfig.from_toml(config_path)

    assert config.repetitions == 3
    assert config.warmups == 1

    with pytest.raises(ConfigError, match="repetitions"):
        BenchmarkConfig(
            encoder_revision=config.encoder_revision,
            llm_revision=config.llm_revision,
            llama_cpp_revision=config.llama_cpp_revision,
            repetitions=0,
        )

    invalid_section = tmp_path / "invalid-section.toml"
    invalid_section.write_text("benchmark = []")
    with pytest.raises(ConfigError, match="section must be an object"):
        BenchmarkConfig.from_toml(invalid_section)


def test_validate_single_token_candidates_rejects_split_labels() -> None:
    assert validate_single_token_candidates({"yes": [11], "no": [12]}) == {
        "yes": 11,
        "no": 12,
    }

    with pytest.raises(
        ValueError,
        match=r"^yes must be represented by a single token$",
    ):
        validate_single_token_candidates({"yes": [11, 13], "no": [12]})


def test_validate_single_token_candidates_uses_shared_candidate_labels(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def labels() -> tuple[str, str]:
        calls.append("called")
        return ("yes", "no")

    monkeypatch.setattr(encoder_module, "candidate_labels", labels)

    assert validate_single_token_candidates({"yes": [11], "no": [12]}) == {
        "yes": 11,
        "no": 12,
    }
    assert calls == ["called"]


def test_mask_position_requires_exactly_one_mask() -> None:
    assert mask_position([3, 99, 4], 99) == 1

    with pytest.raises(
        ValueError,
        match=r"^input must contain exactly one mask token$",
    ):
        mask_position([3, 4], 99)
    with pytest.raises(
        ValueError,
        match=r"^input must contain exactly one mask token$",
    ):
        mask_position([99, 3, 99], 99)


def test_candidate_logprobs_returns_log_softmax_scores() -> None:
    scores = candidate_logprobs(
        logits=[0.0, 2.0, -1.0],
        candidate_token_ids={"yes": 1, "no": 2},
    )

    normalizer = math.log(sum(math.exp(value) for value in (0.0, 2.0, -1.0)))
    assert scores["yes"] == pytest.approx(2.0 - normalizer)
    assert scores["no"] == pytest.approx(-1.0 - normalizer)
    assert candidate_logprobs([1.0], {"yes": 0})["yes"] == 0.0


def test_shared_logsumexp_is_stable_for_large_values() -> None:
    assert logsumexp([1000.0, 1001.0]) == pytest.approx(
        1001.0 + math.log1p(math.exp(-1.0))
    )


def test_shared_finite_number_predicate_rejects_bool_and_nonfinite_values() -> None:
    predicate = getattr(numerics_module, "_is_finite_number", None)

    assert callable(predicate)
    assert predicate(1) is True
    assert predicate(1.5) is True
    assert predicate(True) is False
    assert predicate(math.nan) is False
    assert predicate("1") is False


def test_candidate_logprobs_rejects_empty_nonfinite_and_out_of_range_inputs() -> None:
    with pytest.raises(ValueError, match=r"^logits must not be empty$"):
        candidate_logprobs([], {"yes": 0, "no": 1})
    with pytest.raises(ValueError, match=r"^logits must be finite$"):
        candidate_logprobs([math.nan, 1.0], {"yes": 0, "no": 1})
    with pytest.raises(ValueError, match="out of range"):
        candidate_logprobs([0.0], {"yes": 1, "no": 0})


def test_candidate_variant_logprobs_aggregate_exact_token_ids() -> None:
    scores = candidate_variant_logprobs(
        logits=[0.0, 2.0, -1.0, 1.5, -2.0, 1.0, -3.0],
        candidate_token_ids={"yes": (1, 3, 5), "no": (2, 4, 6)},
    )

    normalizer = math.log(
        sum(math.exp(value) for value in (0.0, 2.0, -1.0, 1.5, -2.0, 1.0, -3.0))
    )
    assert scores["yes"] == pytest.approx(
        math.log(math.exp(2.0) + math.exp(1.5) + math.exp(1.0)) - normalizer
    )
    assert scores["no"] == pytest.approx(
        math.log(math.exp(-1.0) + math.exp(-2.0) + math.exp(-3.0)) - normalizer
    )


def test_validate_single_token_variants_rejects_split_forms() -> None:
    assert validate_single_token_variants({"yes": ((1,), (2,)), "no": ((3,),)}) == {
        "yes": (1, 2),
        "no": (3,),
    }

    with pytest.raises(
        ValueError,
        match=r"^yes must be represented by a single token$",
    ):
        validate_single_token_variants({"yes": ((1, 2),), "no": ((3,),)})
    with pytest.raises(ValueError, match=r"^yes must have a single-token form$"):
        validate_single_token_variants({"yes": (), "no": ((3,),)})
    with pytest.raises(ValueError, match=r"^candidate token IDs must not be empty$"):
        _variant_logprob([0.0], (), 0.0)


def test_encoder_runtime_type_adapters_preserve_typed_inputs() -> None:
    assert _integer_list([1, 2]) == [1, 2]
    assert _float_list([1, 2.5]) == [1.0, 2.5]
    tensor = FakeTensor([[1]])
    assert _tensor_like(tensor) is tensor
    with pytest.raises(ValueError, match=r"^encoder token IDs are invalid$"):
        _integer_list([True])
    with pytest.raises(ValueError, match=r"^encoder token IDs are invalid$"):
        _integer_list("not a list")
    with pytest.raises(ValueError, match=r"^encoder logits are invalid$"):
        _float_list(["not a number"])
    with pytest.raises(ValueError, match=r"^encoder logits are invalid$"):
        _float_list((1.0,))
    with pytest.raises(ValueError, match=r"^encoder tensor value is invalid$"):
        _tensor_like(None)


def test_variant_logprob_rejects_empty_token_ids_with_exact_error() -> None:
    with pytest.raises(ValueError, match=r"^candidate token IDs must not be empty$"):
        _variant_logprob([0.0], (), 0.0)


def test_score_transformers_once_returns_masked_scores_and_timings(monkeypatch) -> None:
    synchronize_calls: list[str] = []
    tokenizer = FakeTokenizer()

    class SynchronizedTorch(FakeTorch):
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(
                synchronize=lambda: synchronize_calls.append("sync")
            )

    ticks = iter(
        (
            10_000_000,
            11_000_000,
            12_000_000,
            13_000_000,
            14_000_000,
            15_000_000,
            16_000_000,
        )
    )
    monkeypatch.setattr(encoder_module.time, "perf_counter_ns", lambda: next(ticks))
    score = score_transformers_once(tokenizer, FakeModel(), SynchronizedTorch(), "cuda")

    assert score.input_tokens == 3
    assert score.logprobs["yes"] > score.logprobs["no"]
    assert score.tokenization_ms == 1.0
    assert score.model_ms == 1.0
    assert score.logprob_ms == 1.0
    assert score.text_to_logprob_ms == 6.0
    assert synchronize_calls == ["sync", "sync"]
    assert tokenizer.calls == [
        (
            encoder_prompt(tokenizer.mask_token),
            {"return_tensors": "pt", "add_special_tokens": True},
        ),
        *[
            (form, {"add_special_tokens": False})
            for label in candidate_labels()
            for form in candidate_label_forms(label)
        ],
    ]
    assert tokenizer.last_encoded is not None
    assert all(
        tensor.to_devices == ["cuda"] for tensor in tokenizer.last_encoded.values()
    )


def test_score_transformers_once_reuses_precomputed_candidate_ids() -> None:
    tokenizer = FakeTokenizer()

    score = score_transformers_once(
        tokenizer,
        FakeModel(),
        FakeTorch(),
        "cpu",
        candidate_token_ids={"yes": (1, 3, 5), "no": (2, 4, 6)},
    )

    assert score.logprobs["yes"] > score.logprobs["no"]
    assert tokenizer.calls == [
        (
            encoder_prompt(tokenizer.mask_token),
            {"return_tensors": "pt", "add_special_tokens": True},
        )
    ]


def test_parse_llama_completion_reads_candidate_logprobs() -> None:
    payload = {
        "completion_probabilities": [
            {
                "probs": [
                    {
                        "token": " yes",
                        "logprob": -0.25,
                        "top_logprobs": [
                            {"token": " no", "logprob": -1.25},
                        ],
                    }
                ]
            }
        ]
    }

    assert parse_candidate_logprobs(payload, ("yes", "no")) == {
        "yes": -0.25,
        "no": -1.25,
    }


def test_parse_candidate_logprobs_reuses_normalized_candidate_map(monkeypatch) -> None:
    seen: list[object] = []

    def capture(
        entry: Mapping[str, object], candidate_by_key: object
    ) -> tuple[str, float] | None:
        assert isinstance(candidate_by_key, dict)
        seen.append(candidate_by_key)
        parsed = llm_module._candidate_token_score(entry)
        if parsed is None:
            return None
        token, score = parsed
        label = candidate_by_key.get(token.strip().casefold())
        return (label, score) if isinstance(label, str) else None

    monkeypatch.setattr(llm_module, "_candidate_logprob", capture)
    payload = {
        "completion_probabilities": [
            {
                "probs": [
                    {
                        "token": "yes",
                        "logprob": -0.25,
                        "top_logprobs": [
                            {"token": "no", "logprob": -1.25},
                        ],
                    }
                ]
            }
        ]
    }

    assert parse_candidate_logprobs(payload, ("yes", "no")) == {
        "yes": -0.25,
        "no": -1.25,
    }
    assert len(seen) == 2
    assert seen[0] is seen[1]


def test_parse_llama_aggregates_case_and_spacing_variants_without_overwriting() -> None:
    payload = {
        "completion_probabilities": [
            {
                "probs": [
                    {
                        "token": " Yes",
                        "logprob": -0.25,
                        "top_logprobs": [
                            {"token": "yes", "logprob": -1.25},
                            {"token": " no", "logprob": -2.0},
                            {"token": "No", "logprob": -3.0},
                        ],
                    }
                ]
            }
        ]
    }

    assert parse_candidate_logprobs(payload, ("yes", "no")) == {
        "yes": pytest.approx(math.log(math.exp(-0.25) + math.exp(-1.25))),
        "no": pytest.approx(math.log(math.exp(-2.0) + math.exp(-3.0))),
    }


def test_parse_llama_current_completion_shape_reads_top_logprobs() -> None:
    payload = {
        "completion_probabilities": [
            {
                "token": "\\n\\n",
                "logprob": -0.05,
                "top_logprobs": [
                    {"token": " yes", "logprob": -3.0},
                    {"token": " no", "logprob": -8.0},
                ],
            }
        ]
    }

    assert parse_candidate_logprobs(payload, ("yes", "no")) == {
        "yes": -3.0,
        "no": -8.0,
    }


def test_parse_llama_completion_rejects_missing_candidate() -> None:
    with pytest.raises(
        LlamaResponseError,
        match=r"^response is missing candidate logprobs: yes, no$",
    ):
        parse_candidate_logprobs(
            {
                "completion_probabilities": [
                    {"probs": [{"token": "maybe", "logprob": -1.0}]}
                ]
            },
            ("yes", "no"),
        )


def test_parse_llama_openai_choices_reads_candidate_logprobs() -> None:
    payload = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "yes",
                            "logprob": -0.25,
                            "top_logprobs": [{"token": "no", "logprob": -1.25}],
                        }
                    ]
                }
            }
        ]
    }

    assert parse_candidate_logprobs(payload, ("yes", "no")) == {
        "yes": -0.25,
        "no": -1.25,
    }


def test_parse_llama_skips_invalid_and_non_candidate_entries() -> None:
    payload = {
        "completion_probabilities": [
            {
                "probs": [
                    {
                        "token": "yes",
                        "logprob": -0.25,
                        "top_logprobs": [
                            {"token": None, "logprob": -2.0},
                            {"token": "maybe", "logprob": -2.0},
                            {"token": "yes", "logprob": float("nan")},
                            {"token": "no", "logprob": -1.25},
                            {"token": "invalid", "logprob": "not-a-number"},
                        ],
                    }
                ]
            }
        ]
    }

    assert parse_candidate_logprobs(payload, ("yes", "no")) == {
        "yes": -0.25,
        "no": -1.25,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"completion_probabilities": []},
        {"completion_probabilities": [None]},
        {"completion_probabilities": [{"probs": "invalid"}]},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"logprobs": {}}]},
        {"choices": [{"logprobs": {"content": "invalid"}}]},
    ],
)
def test_parse_llama_rejects_missing_probability_shapes(payload) -> None:
    with pytest.raises(
        LlamaResponseError,
        match=r"^response contains no completion logprobs$",
    ):
        parse_candidate_logprobs(payload, ("yes", "no"))


def test_llama_response_helpers_reject_malformed_values(monkeypatch) -> None:
    assert _timing_ms({}) is None
    assert _timing_ms({"timings": {"prompt_ms": 1.0}}) is None
    assert _timing_ms({"timings": {"predicted_ms": 1.0}}) is None
    assert _timing_ms({"timings": {"prompt_ms": 0.0, "predicted_ms": 0.0}}) == 0.0
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp timings are not numeric$",
    ):
        _timing_ms({"timings": {"prompt_ms": "slow"}})
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp timings are not numeric$",
    ):
        _timing_ms({"timings": {"prompt_ms": True}})
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp timings are invalid$",
    ):
        _timing_ms({"timings": {"prompt_ms": -1.0, "predicted_ms": 1.0}})
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp timings are invalid$",
    ):
        _timing_ms(
            {
                "timings": {
                    "prompt_ms": sys.float_info.max,
                    "predicted_ms": sys.float_info.max,
                }
            }
        )
    assert _input_tokens({}) is None
    assert _input_tokens({"usage": {}}) is None
    assert _input_tokens({"usage": {"prompt_tokens": 0}}) == 0
    assert _input_tokens({"tokens_evaluated": 18}) == 18
    assert _input_tokens({"timings": {"prompt_n": 19}}) == 19
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp prompt token count is invalid$",
    ):
        _input_tokens({"usage": {"prompt_tokens": -1}})
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp prompt token count is invalid$",
    ):
        _input_tokens({"usage": {"prompt_tokens": True}})
    assert _with_top_logprobs([{"token": "yes"}]) == [{"token": "yes"}]
    with pytest.raises(
        LlamaResponseError,
        match=r"^token probability entry is not an object$",
    ):
        _with_top_logprobs([None])

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    monkeypatch.setattr(llm_module, "urlopen", lambda *_args, **_kwargs: Response())
    assert _post_json("http://example.test", {"x": 1}, 1.0) == {"ok": True}

    class InvalidResponse(Response):
        def read(self):
            return b"[]"

    monkeypatch.setattr(
        llm_module, "urlopen", lambda *_args, **_kwargs: InvalidResponse()
    )
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp response is not an object$",
    ):
        _post_json("http://example.test", {"x": 1}, 1.0)


def test_llama_wire_contract_uses_explicit_utf8_and_header_spelling(
    monkeypatch,
) -> None:
    EncodedText.encodings = []
    EncodedBytes.decodings = []
    captured_requests: list[CapturedRequest] = []
    monkeypatch.setattr(llm_module.json, "dumps", lambda _payload: EncodedText("{}"))
    monkeypatch.setattr(llm_module, "Request", CapturedRequest)
    monkeypatch.setattr(
        llm_module,
        "urlopen",
        lambda request, **_kwargs: (
            captured_requests.append(request) or EncodedJsonResponse()
        ),
    )

    assert _post_json("http://example.test", {"x": 1}, 1.0) == {"ok": True}
    assert EncodedText.encodings == ["utf-8"]
    assert EncodedBytes.decodings == ["utf-8"]
    assert captured_requests[0].values["headers"] == {
        "Content-Type": "application/json"
    }


def test_llama_response_helper_error_messages_are_stable() -> None:
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp template response has no rendered prompt$",
    ):
        llm_module._rendered_prompt({"prompt": 1})
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp timings are not numeric$",
    ):
        _timing_value({"prompt_ms": "slow"}, "prompt_ms")
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp timings are invalid$",
    ):
        _timing_value({"prompt_ms": -1.0}, "prompt_ms")
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp timings are invalid$",
    ):
        _timing_total(sys.float_info.max, sys.float_info.max)
    with pytest.raises(
        LlamaResponseError,
        match=r"^llama\.cpp prompt token count is invalid$",
    ):
        _validated_token_count("18")
    with pytest.raises(
        LlamaResponseError,
        match=r"^response contains no completion logprobs$",
    ):
        _entries({})
    with pytest.raises(
        LlamaResponseError,
        match=r"^response contains no token probabilities$",
    ):
        _first_logprob_entry([])
    with pytest.raises(
        LlamaResponseError,
        match=r"^token probability entry is not an object$",
    ):
        _first_logprob_entry([None])
    with pytest.raises(
        LlamaResponseError,
        match=r"^response is missing candidate logprobs: no$",
    ):
        _require_candidate_scores({"yes": [-1.0]}, ("yes", "no"))
    with pytest.raises(
        LlamaResponseError,
        match=r"^response is missing candidate logprobs: yes, no$",
    ):
        _require_candidate_scores({}, ("yes", "no"))
    with pytest.raises(
        LlamaResponseError,
        match=r"^candidate logprobs must not be empty$",
    ):
        _logsumexp([])


def test_post_json_preserves_request_contract(monkeypatch) -> None:
    captured_request: Request | None = None
    captured_timeout: float | None = None

    def open_request(request: Request, timeout: float) -> JsonResponse:
        nonlocal captured_request, captured_timeout
        captured_request = request
        captured_timeout = timeout
        return JsonResponse()

    monkeypatch.setattr(llm_module, "urlopen", open_request)
    assert _post_json("http://example.test", {"x": 1}, 1.0) == {"ok": True}
    assert captured_request is not None
    request = captured_request
    assert request.full_url == "http://example.test"
    assert request.data == b'{"x": 1}'
    assert request.get_header("Content-type") == "application/json"
    assert captured_timeout == 1.0


def test_llama_client_returns_scores_and_server_timing(monkeypatch) -> None:
    requests = []

    def request(url, payload, timeout):
        requests.append((url, payload, timeout))
        if url.endswith("/apply-template"):
            return {"prompt": "rendered prompt"}
        return {
            "completion_probabilities": [
                {
                    "probs": [
                        {
                            "token": " yes",
                            "logprob": -0.25,
                            "top_logprobs": [{"token": " no", "logprob": -1.25}],
                        }
                    ]
                }
            ],
            "timings": {"prompt_ms": 2.0, "predicted_ms": 0.5},
            "usage": {"prompt_tokens": 18},
        }

    ticks = iter((10_000_000, 11_000_000, 12_000_000, 13_000_000))
    monkeypatch.setattr(llm_module.time, "perf_counter_ns", lambda: next(ticks))
    score = LlamaClient("http://127.0.0.1:8080/X///", request=request).score(seed=7)

    assert score.logprobs == {"yes": -0.25, "no": -1.25}
    assert score.tokenization_ms is None
    assert score.model_ms == pytest.approx(2.5)
    assert score.logprob_ms == 1.0
    assert score.text_to_logprob_ms == 3.0
    assert score.input_tokens == 18
    assert requests[0][0] == "http://127.0.0.1:8080/X/apply-template"
    assert requests[0][1] == chat_template_request_payload(llm_prompt())
    assert requests[0][1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert requests[0][2] == 300.0
    assert requests[1][0] == "http://127.0.0.1:8080/X/completion"
    assert requests[1][2] == 300.0
    assert requests[1][1]["prompt"] == "rendered prompt"
    assert requests[1][1]["seed"] == 7


def test_chat_template_payload_contains_only_the_fixed_user_message() -> None:
    assert chat_template_request_payload("Question") == {
        "messages": [{"role": "user", "content": "Question"}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_completion_request_uses_one_prediction_without_prompt_cache() -> None:
    payload = completion_request_payload("Answer:", seed=7)

    assert payload == {
        "prompt": "Answer:",
        "n_predict": 1,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "seed": 7,
        "cache_prompt": False,
        "n_probs": 32,
        "timings_per_token": True,
        "stream": False,
    }


def test_llama_score_marks_unmeasured_timings_as_none() -> None:
    client = LlamaClient(
        "http://test",
        request=lambda url, _payload, _timeout: (
            {"prompt": "rendered prompt"}
            if url.endswith("/apply-template")
            else {
                "completion_probabilities": [
                    {
                        "probs": [
                            {
                                "token": "yes",
                                "logprob": -0.25,
                                "top_logprobs": [{"token": "no", "logprob": -1.25}],
                            }
                        ]
                    }
                ]
            }
        ),
    )

    score = client.score(seed=7)

    assert score.tokenization_ms is None
    assert score.model_ms is None
    assert score.logprob_ms >= 0
    assert score.text_to_logprob_ms >= score.logprob_ms
