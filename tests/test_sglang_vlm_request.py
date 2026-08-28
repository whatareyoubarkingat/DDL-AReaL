from areal.api import ModelRequest
from areal.engine.sglang_remote import SGLangBackend


class Qwen2_5_VLProcessor:
    image_token_id = 151655


def test_vlm_request_collapses_expanded_image_token_runs():
    image_token_id = Qwen2_5_VLProcessor.image_token_id
    original = [1, image_token_id, image_token_id, 2, image_token_id, image_token_id, 3]
    req = ModelRequest(
        input_ids=original.copy(),
        image_data=["image-a", "image-b"],
        processor=Qwen2_5_VLProcessor(),
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


def test_text_request_keeps_input_ids_unchanged():
    req = ModelRequest(input_ids=[1, 2, 2, 3], image_data=None)

    request = SGLangBackend().build_generation_request(req, with_lora=False, version=0)

    assert request.payload["input_ids"] == req.input_ids


def test_vlm_request_without_processor_keeps_input_ids_unchanged():
    req = ModelRequest(input_ids=[1, 2, 2, 3], image_data=["image"])

    request = SGLangBackend().build_generation_request(req, with_lora=False, version=0)

    assert request.payload["input_ids"] == req.input_ids
