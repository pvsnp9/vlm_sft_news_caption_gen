import os
import sys
from typing import Any, Dict, List

import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from collators.qwen3_collator import Qwen3VisionSFTCollator
from dataset.loaddataset import load_xl_munichus
from config.logistics import Logistics, ModelCards, SYS_PROMPT


def _load_processor():
    mc = ModelCards()
    try:
        processor = AutoProcessor.from_pretrained(mc.qwen3_vl_8b_instruct, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Processor load failed: {exc}")

    print("Loaded processor:", mc.qwen3_vl_8b_instruct)
    print(
        "Tokenizer padding token:",
        processor.tokenizer.pad_token,
        "| pad_id:",
        processor.tokenizer.pad_token_id,
    )
    print("Model max length:", getattr(processor.tokenizer, "model_max_length", "unknown"))
    print(f"vocab_size={len(processor.tokenizer)}")
    print(f"padding_side={processor.tokenizer.padding_side}")
    print(f"special_tokens_map={processor.tokenizer.special_tokens_map}")
    print(f"additional_special_tokens={processor.tokenizer.additional_special_tokens}")
    return processor


def _load_samples():
    logistics = Logistics()
    dataset_id = logistics.hf_datatset_id
    try:
        ds, _langs = load_xl_munichus("train", lang="eng")
    except Exception as exc:
        pytest.skip(f"Dataset load failed: {exc}")

    samples = [ds[i] for i in range(min(3, len(ds)))]
    print(f"dataset_id={dataset_id}")
    print(f"samples_loaded={len(samples)}")
    print(f"system_prompt={SYS_PROMPT}")
    return samples


def _synthetic_batch() -> List[Dict[str, Any]]:
    return [
        {
            "image": Image.new("RGB", (16, 16), color=(0, 128, 255)),
            "caption": "A speaker addresses a crowd.",
            "content": "The mayor spoke at a downtown rally to discuss public safety measures.",
            "language": "en",
        }
    ]


@pytest.fixture(scope="session")
def qwen_processor():
    return _load_processor()


@pytest.fixture(scope="session")
def qwen_samples():
    return _load_samples()


def _map_example(example: Dict[str, Any]) -> Dict[str, Any]:
    language = example.get("language")
    if not language:
        language = example.get("language_code") or "eng"
    return {
        "image": example.get("image"),
        "caption": example.get("caption", ""),
        "content": example.get("content", ""),
        "language": language,
    }


def test_init_params(qwen_processor):
    collator = Qwen3VisionSFTCollator(
        processor=qwen_processor,
        max_length=512,
        add_generation_prompt=False,
        training=True,
    )
    assert collator.max_length == 512
    assert collator.add_generation_prompt is False
    assert collator.training is True

    collator_eval = Qwen3VisionSFTCollator(
        processor=qwen_processor,
        max_length=256,
        add_generation_prompt=True,
        training=False,
    )
    assert collator_eval.max_length == 256
    assert collator_eval.add_generation_prompt is True
    assert collator_eval.training is False


def test_normalize_image(qwen_processor):
    collator = Qwen3VisionSFTCollator(processor=qwen_processor, training=True)
    valid = Image.new("RGB", (16, 16), color=(255, 0, 0))
    out = collator._normalize_image(valid)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == (16, 16)

    invalid = None
    out_invalid = collator._normalize_image(invalid)
    assert isinstance(out_invalid, Image.Image)
    assert out_invalid.mode == "RGB"
    assert out_invalid.size == (1, 1)


def test_build_messages(qwen_processor, qwen_samples):
    collator = Qwen3VisionSFTCollator(processor=qwen_processor, training=True)
    ex = _map_example(qwen_samples[0])
    messages = collator._build_messages_from_example(ex)
    print("messages:")
    print("```text")
    print(messages)
    print("```")
    chat_text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    print("chat_format:")
    print("```text")
    print(chat_text)
    print("```")

    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_call_training_and_eval(qwen_processor, qwen_samples):
    batch = [_map_example(s) for s in qwen_samples[:2]]
    collator = Qwen3VisionSFTCollator(processor=qwen_processor, training=True, max_length=1500)

    train_outputs = collator(batch)
    assert "input_ids" in train_outputs
    assert "labels" in train_outputs
    assert train_outputs["input_ids"].shape == train_outputs["labels"].shape

    labels = train_outputs["labels"]
    input_ids = train_outputs["input_ids"]
    pad_id = qwen_processor.tokenizer.pad_token_id
    if pad_id is not None:
        assert torch.all(labels[input_ids == pad_id] == -100)
    assert torch.any(labels != -100)
    assert torch.any(labels == -100)

    grid = train_outputs.get("image_grid_thw")
    if grid is not None:
        grid_list = grid.tolist()
        token_counts = [int(t * h * w) for t, h, w in grid_list]
        print("image_grid_thw:", grid_list)
        print("vision_token_counts:", token_counts)
    else:
        pixel_values = train_outputs.get("pixel_values")
        if pixel_values is not None:
            print("pixel_values.shape:", tuple(pixel_values.shape))

    prompt_texts: List[str] = []
    for example in batch:
        messages = collator._build_messages_from_example(example)
        prompt_texts.append(
            qwen_processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
        )
    prompt_inputs = qwen_processor(
        text=prompt_texts,
        images=[collator._normalize_image(b["image"]) for b in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=collator.max_length,
    )
    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
    for i, plen in enumerate(prompt_lens):
        assert torch.all(labels[i, : int(plen)] == -100)

    for i in range(min(2, input_ids.shape[0])):
        decoded_input = qwen_processor.tokenizer.decode(
            input_ids[i].tolist(), skip_special_tokens=False
        )
        label_ids = [t for t in labels[i].tolist() if t != -100]
        decoded_labels = qwen_processor.tokenizer.decode(
            label_ids, skip_special_tokens=False
        )
        print(f"decoded_input[{i}]:")
        print("```text")
        print(decoded_input)
        print("```")
        print(f"decoded_labels[{i}]:")
        print("```text")
        print(decoded_labels)
        print("```")

    collator_eval = Qwen3VisionSFTCollator(processor=qwen_processor, training=False, max_length=1500)
    infer_outputs = collator_eval(batch)
    assert "labels" not in infer_outputs, "Inference outputs should not include labels."
    assert "input_ids" in infer_outputs, "Inference outputs should include input_ids."


def test_synthetic_batch_has_supervised_tokens(qwen_processor):
    batch = _synthetic_batch()
    collator = Qwen3VisionSFTCollator(processor=qwen_processor, training=True, max_length=512)
    outputs = collator(batch)
    assert "labels" in outputs
    assert torch.any(outputs["labels"] != -100)


def test_dataset_smoke_has_supervised_tokens(qwen_processor):
    if os.environ.get("QWEN3_DATASET_SMOKE") != "1":
        pytest.skip("Set QWEN3_DATASET_SMOKE=1 to enable dataset smoke test.")
    try:
        ds, _langs = load_xl_munichus("train", lang="eng")
    except Exception as exc:
        pytest.skip(f"Dataset load failed: {exc}")
    if len(ds) == 0:
        pytest.skip("Dataset empty.")
    batch = [_map_example(ds[i]) for i in range(min(2, len(ds)))]
    collator = Qwen3VisionSFTCollator(processor=qwen_processor, training=True, max_length=512)
    outputs = collator(batch)
    assert "labels" in outputs
    assert torch.any(outputs["labels"] != -100)


def _run_visual_inspection() -> None:
    processor = _load_processor()
    samples = _load_samples()
    test_init_params(processor)
    test_normalize_image(processor)
    test_build_messages(processor, samples)
    test_call_training_and_eval(processor, samples)
    test_synthetic_batch_has_supervised_tokens(processor)
    if os.environ.get("QWEN3_DATASET_SMOKE") == "1":
        test_dataset_smoke_has_supervised_tokens(processor)


if __name__ == "__main__":
    _run_visual_inspection()
