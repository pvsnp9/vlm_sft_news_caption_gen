from __future__ import annotations

from typing import Any, Dict, List

import torch
from PIL import Image

from config.logistics import PROMPT_TPML_V1, SYS_PROMPT


class AyaVisionSFTCollator:
    def __init__(
        self,
        processor: Any,
        max_length: int = 2048,
        add_generation_prompt: bool = True,
        training: bool = True,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.add_generation_prompt = add_generation_prompt
        self.training = training

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            if self.training:
                tokenizer.padding_side = "right"
                tokenizer.truncation_side = "right"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

    def _normalize_image(self, image: Any) -> Image.Image:
        try:
            if isinstance(image, Image.Image) and getattr(image, "mode", None):
                pil = image
            else:
                pil = Image.fromarray(image)
            pil = pil.convert("RGB")
            if not pil.size or pil.size[0] <= 0 or pil.size[1] <= 0:
                raise ValueError("invalid image size")
            return pil
        except Exception:
            return Image.new("RGB", (1, 1))

    def _build_messages_from_example(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            caption = str(example.get("caption") or "").strip()
            content = str(example.get("content") or "").strip()
            language = str(example.get("language") or "").strip() or "en"
            news = content[:1200].replace("{", "{{").replace("}", "}}")
            prompt = PROMPT_TPML_V1.format(news=news, language=language)
        except Exception:
            caption = str(example.get("caption") or "").strip()
            language = str(example.get("language") or "").strip() or "en"
            prompt = PROMPT_TPML_V1.format(news="", language=language)

        return [
            {"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": caption}]},
        ]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        try:
            if not batch:
                raise ValueError("empty batch")

            images: List[Image.Image] = []
            full_texts: List[str] = []
            prompt_texts: List[str] = []

            for example in batch:
                images.append(self._normalize_image(example.get("image")))
                messages = self._build_messages_from_example(example)

                if self.training:
                    full_texts.append(
                        self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                    )
                    prompt_texts.append(
                        self.processor.apply_chat_template(
                            messages[:-1], tokenize=False, add_generation_prompt=True
                        )
                    )
                else:
                    full_texts.append(
                        self.processor.apply_chat_template(
                            messages[:-1],
                            tokenize=False,
                            add_generation_prompt=self.add_generation_prompt,
                            add_generic_prompt=True,
                        )
                    )

            model_inputs = self.processor(
                text=full_texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            if not self.training:
                return model_inputs

            prompt_inputs = self.processor(
                text=prompt_texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()
            full_lens = model_inputs["attention_mask"].sum(dim=1).tolist()

            labels = model_inputs["input_ids"].clone()
            for i, prompt_len in enumerate(prompt_lens):
                prompt_len = int(prompt_len)
                if prompt_len > 0:
                    labels[i, :prompt_len] = -100
                full_len = int(full_lens[i])
                if full_len > prompt_len:
                    labels[i, prompt_len:full_len] = model_inputs["input_ids"][
                        i, prompt_len:full_len
                    ]

            pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                labels[model_inputs["input_ids"] == pad_id] = -100

            keep_mask = (labels != -100).any(dim=1)
            if keep_mask.any():
                for k, v in list(model_inputs.items()):
                    if isinstance(v, torch.Tensor) and v.size(0) == keep_mask.size(0):
                        model_inputs[k] = v[keep_mask]
                labels = labels[keep_mask]

            model_inputs["labels"] = labels
            return model_inputs
        except Exception as exc:
            raise RuntimeError(f"Aya collate failed: {exc}") from exc
