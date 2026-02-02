from typing import List, Dict, Any

import torch
from PIL import Image

from config.logistics import PROMPT_TPML_V1, SYS_PROMPT


def normalize_image(img: Any) -> Image.Image:
    try:
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img)
            except Exception:
                img = Image.new("RGB", (1, 1))
        img = img.convert("RGB")
        if img.size[0] <= 0 or img.size[1] <= 0:
            return Image.new("RGB", (1, 1))
        max_side = max(img.size)
        if max_side > 1120:
            img.thumbnail((1120, 1120))
        return img
    except Exception:
        return Image.new("RGB", (1, 1))


def build_messages_from_example(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        caption = str(ex.get("caption", "")).strip()
        content = str(ex.get("content", "")).strip()[:1200]
        language = str(ex.get("language", "English")).strip() or "English"
        user_text = PROMPT_TPML_V1.format(news=content, language=language)
        return [
            {
                "role": "system",
                "content": SYS_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            },
        ]
    except Exception:
        caption = str(ex.get("caption", ""))
        content = str(ex.get("content", ""))
        language = str(ex.get("language", "English")) or "English"
        user_text = PROMPT_TPML_V1.format(news=content[:1200], language=language)
        return [
            {
                "role": "system",
                "content": SYS_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            },
        ]


class Llama32SFTCollator:
    def __init__(
        self,
        processor,
        max_length: int = 1500,
        add_generation_prompt: bool = True,
        training: bool = True,
    ):
        try:
            self.processor = processor
            self.max_length = max_length
            self.add_generation_prompt = add_generation_prompt
            self.training = training
            if self.training:
                self.processor.tokenizer.padding_side = "right"
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        except Exception as exc:
            raise RuntimeError(f"Llama32SFTCollator init failed: {exc}") from exc

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            images = [[normalize_image(ex["image"])] for ex in batch]
            messages_list = [build_messages_from_example(ex) for ex in batch]

            if self.training:
                full_texts = self.processor.apply_chat_template(
                    messages_list, tokenize=False, add_generation_prompt=False
                )
                prompt_texts = self.processor.apply_chat_template(
                    [m[:-1] for m in messages_list],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                try:
                    enc = self.processor(
                        text=full_texts,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        text_kwargs={"add_special_tokens": False},
                    )
                except Exception:
                    enc = self.processor(
                        text=full_texts,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                    )

                try:
                    prompt_enc = self.processor(
                        text=prompt_texts,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        text_kwargs={"add_special_tokens": False},
                    )
                except Exception:
                    prompt_enc = self.processor(
                        text=prompt_texts,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                    )

                prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
                labels = enc["input_ids"].clone()
                for i in range(labels.size(0)):
                    labels[i, : prompt_lens[i].item()] = -100
                pad_id = self.processor.tokenizer.pad_token_id
                labels[labels == pad_id] = -100
                keep_mask = (labels != -100).any(dim=1)
                if keep_mask.any():
                    for k, v in list(enc.items()):
                        if isinstance(v, torch.Tensor) and v.size(0) == keep_mask.size(0):
                            enc[k] = v[keep_mask]
                    labels = labels[keep_mask]
                enc["labels"] = labels
                return enc

            input_msgs = [m[:-1] for m in messages_list]
            texts = self.processor.apply_chat_template(
                input_msgs,
                tokenize=False,
                add_generation_prompt=self.add_generation_prompt,
            )
            try:
                enc = self.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    text_kwargs={"add_special_tokens": False},
                )
            except Exception:
                enc = self.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
            return enc
        except Exception as exc:
            raise RuntimeError(f"Llama32SFTCollator failed: {exc}") from exc
