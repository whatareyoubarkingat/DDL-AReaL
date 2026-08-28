import pytest

from areal.api import ModelRequest
from areal.engine.sglang_remote import SGLangBackend


class CapabilityBasedVisionProcessor:
    image_token_id = 151655


def test_vlm_request_with_image_token_capability_collapses_expanded_runs():
    """Normalize expanded placeholders without depending on a class name."""
    image_token_id = CapabilityBasedVisionProcessor.image_token_id
    original = [1, image_token_id, image_token_id, 2, image_token_id, image_token_id, 3]
    req = ModelRequest(
        input_ids=original.copy(),
        image_data=["image-a", "image-b"],
        processor=CapabilityBasedVisionProcessor(),
    )

    request = SGLangBackend().build_generation_request(req, with_lora=False, version=0)

    assert request.payload["input_ids"] == [
        1,
        image_token_id,
        2,
        image_token_id,
        3,
    ]
    assert req.input_ids == original


def test_vlm_request_resolves_image_token_with_processor_tokenizer():
    """Support processors that expose the image token but not its ID."""

    class Tokenizer:
        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            assert token == "<image>"
            return 42

    class Processor:
        image_token = "<image>"
        tokenizer = Tokenizer()

    req = ModelRequest(
        input_ids=[1, 42, 42, 2],
        image_data=["image"],
        processor=Processor(),
    )

    request = SGLangBackend().build_generation_request(req, with_lora=False, version=0)

    assert request.payload["input_ids"] == [1, 42, 2]


def test_vlm_request_rejects_mismatched_image_placeholders():
    """Fail before SGLang when placeholders and image payloads disagree."""
    req = ModelRequest(
        input_ids=[1, CapabilityBasedVisionProcessor.image_token_id, 2],
        image_data=["image-a", "image-b"],
        processor=CapabilityBasedVisionProcessor(),
    )

    with pytest.raises(ValueError, match="image placeholders do not match image_data"):
        SGLangBackend().build_generation_request(req, with_lora=False, version=0)


def test_vlm_request_without_image_token_capability_keeps_input_ids():
    """Leave processors with a different placeholder contract untouched."""

    class OtherVisionProcessor:
        pass

    req = ModelRequest(
        input_ids=[1, 2, 2, 3],
        image_data=["image"],
        processor=OtherVisionProcessor(),
    )

    request = SGLangBackend().build_generation_request(req, with_lora=False, version=0)

    assert request.payload["input_ids"] == req.input_ids


def test_text_request_keeps_input_ids_unchanged():
    req = ModelRequest(input_ids=[1, 2, 2, 3], image_data=None)

    request = SGLangBackend().build_generation_request(req, with_lora=False, version=0)

    assert request.payload["input_ids"] == req.input_ids


def test_vlm_request_without_processor_warns_and_keeps_input_ids():
    """Surface malformed multimodal requests instead of failing silently."""
    req = ModelRequest(input_ids=[1, 2, 2, 3], image_data=["image"])

    with pytest.warns(RuntimeWarning, match="without a processor"):
        request = SGLangBackend().build_generation_request(
            req, with_lora=False, version=0
        )

    assert request.payload["input_ids"] == req.input_ids
