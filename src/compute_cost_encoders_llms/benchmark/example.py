from __future__ import annotations

LANDUSE_SENTENCE = (
    "A public park with grass, trees, and walking paths occupies the parcel."
)
LANDUSE_QUESTION = "Is this sentence relevant for a land use description?"


def candidate_labels() -> tuple[str, str]:
    """Return the fixed binary decision labels in stable order."""

    return ("yes", "no")


def encoder_prompt(mask_token: str) -> str:
    """Build the masked-token prompt for the encoder."""

    if not mask_token:
        raise ValueError("mask_token must not be empty")
    return (
        f'Here is a target sentence: "{LANDUSE_SENTENCE}"\n'
        f"{LANDUSE_QUESTION} {mask_token}"
    )


def llm_prompt() -> str:
    """Build the next-token prompt for the causal LLM."""

    return (
        f'Here is a target sentence: "{LANDUSE_SENTENCE}"\n'
        f"{LANDUSE_QUESTION} "
        "Answer with exactly yes or no.\nAnswer:"
    )
