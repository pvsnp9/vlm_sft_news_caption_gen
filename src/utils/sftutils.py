from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import os
import wandb
from transformers.trainer_callback import TrainerCallback
from config.logistics import Logistics



class LogCallback(TrainerCallback):
    def __init__(self, logging_steps: int) -> None:
        self.logging_steps = logging_steps

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        if state.global_step % self.logging_steps == 0:
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            print(f"step={state.global_step} loss={loss} lr={lr}")


class EvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_fn,
        run: Optional[wandb.sdk.wandb_run.Run],
        eval_cfg: Dict[str, Any],
        eval_dataset: Any,
        eval_collator: Any,
        processor: Any,
    ) -> None:
        self.eval_fn = eval_fn
        self.run = run
        self.eval_cfg = eval_cfg
        self.eval_dataset = eval_dataset
        self.eval_collator = eval_collator
        self.processor = processor

    def on_evaluate(self, args, state, control, **kwargs):  # type: ignore[override]
        model = kwargs.get("model")
        if model is None:
            return
        try:
            metrics = self.eval_fn(
                model, self.processor, self.eval_dataset, self.eval_collator, self.eval_cfg
            )
            if self.run is not None:
                self.run.log({f"gen_eval/{k}": v for k, v in metrics.items()})
        except Exception as exc:
            print(f"Generation eval skipped: {exc}")

def get_sft_result_dir(model):
    path = os.path.join(Logistics().results_dir, model)
    os.makedirs(path, exist_ok=True)
    return path


def _dtype_from_str(dtype: str):
    import torch

    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return torch.float32


def _select_subset(dataset: Any, max_samples: Optional[int]) -> Any:
    if not max_samples:
        return dataset
    if hasattr(dataset, "select"):
        return dataset.select(range(min(len(dataset), max_samples)))
    return list(dataset.take(max_samples))


def _dataset_size(dataset: Any) -> Optional[int]:
    try:
        return len(dataset)
    except Exception:
        return None


def _count_trainable_params(model) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params": total}


def _get_world_size() -> int:
    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _build_bnb_config(cfg: Dict[str, Any]):
    from transformers import BitsAndBytesConfig

    qlora = cfg["qlora"]
    return BitsAndBytesConfig(
        load_in_4bit=qlora.load_in_4bit,
        bnb_4bit_use_double_quant=qlora.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=qlora.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_dtype_from_str(qlora.bnb_4bit_compute_dtype),
    )


def _build_lora_config(cfg: Dict[str, Any]):
    from peft import LoraConfig

    lora = cfg["lora"]
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        target_modules=lora.target_modules,
        task_type="CAUSAL_LM",
    )


def _resolve_adapter_dir(cfg: Dict[str, Any]) -> str:
    adapter_dir = cfg["output"].get("adapter_output_dir")
    if adapter_dir:
        return adapter_dir
    return os.path.join(cfg["logistics"].models_output_dir, cfg["output"]["model_dir"])


def _verify_adapter_artifacts(adapter_output_dir: str) -> None:
    if not os.path.exists(os.path.join(adapter_output_dir, "adapter_config.json")):
        raise RuntimeError("adapter_config.json not found after training")
    has_model = any(
        name.startswith("adapter_model")
        for name in os.listdir(adapter_output_dir)
        if os.path.isfile(os.path.join(adapter_output_dir, name))
    )
    if not has_model:
        raise RuntimeError("adapter_model weights not found after training")
