from __future__ import annotations

LANDUSE_SENTENCE = (
    "A public park with grass, trees, and walking paths occupies the parcel."
)


def candidate_labels() -> tuple[str, str]:
    """Return the fixed binary decision labels in stable order."""

    return ("yes", "no")


def encoder_prompt(mask_token: str) -> str:
    """Build the masked-token prompt for the encoder."""

    if not mask_token:
        raise ValueError("mask_token must not be empty")
    return f"Sentence: {LANDUSE_SENTENCE}\nLand-use sentence? {mask_token}"


def llm_prompt() -> str:
    """Build the next-token prompt for the causal LLM."""

    return (
        "Decide whether the following sentence describes land use. "
        "Answer with exactly yes or no.\n"
        f"Sentence: {LANDUSE_SENTENCE}\nAnswer:"
    )
