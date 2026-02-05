import json
import os
from datetime import datetime

from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from config.logistics import ModelCards, build_cfg
from src.collators.llama32_collator import Llama32SFTCollator
from src.dataset.loaddataset import load_train_eval_sft_dataset
from src.utils.logging import init_wandb, log_run_metadata
from src.utils.sftutils import (
    LogCallback,
    _build_bnb_config,
    _build_lora_config,
    _count_trainable_params,
    _dataset_size,
    _dtype_from_str,
    _get_world_size,
    _resolve_adapter_dir,
    _select_subset,
    _verify_adapter_artifacts,
)


def main() -> None:
    cfg = build_cfg(ModelCards().llama3_2vl)
    logistics = cfg["logistics"]
    cfg["sft"].max_length = ModelCards().llama_max_length
    cfg["sft_extras"].use_flash_attention_2 = False

    cfg["sft"].batch_size = 2
    cfg["sft"].gradient_accumulation_steps = 16 #64 batch

    sp = cfg["sft"]
    extras = cfg["sft_extras"]

    set_seed(sp.seed)

    adapter_output_dir = _resolve_adapter_dir(cfg)
    os.makedirs(adapter_output_dir, exist_ok=True)
    os.makedirs(logistics.sft_log_dir, exist_ok=True)

    run = init_wandb(cfg.get("wandb", {}), cfg["output"]["run_name"])

    model_name_or_path = cfg["model"]["base_model_name_or_path"]
    cache_dir = logistics.hf_cache_dir

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        cache_dir=cache_dir,
    )
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    bnb_config = _build_bnb_config(cfg)
    torch_dtype = _dtype_from_str(cfg["model"].get("torch_dtype", "float16"))
    use_flash = bool(getattr(extras, "use_flash_attention_2", False))
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "cache_dir": cache_dir,
    }
    if use_flash:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name_or_path,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except Exception:
            use_flash = False
            model = AutoModelForImageTextToText.from_pretrained(
                model_name_or_path, **model_kwargs
            )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path, **model_kwargs
        )

    if cfg["model"]["gradient_checkpointing"] or extras.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, _build_lora_config(cfg))
    if getattr(model, "config", None) is not None:
        model.config.use_cache = extras.use_cache

    train_datasets, eval_dataset = load_train_eval_sft_dataset(sample_size=logistics.train_sample_size)
    train_datasets = _select_subset(train_datasets, cfg["dataset"]["max_train_samples"])
    eval_dataset = _select_subset(eval_dataset, cfg["dataset"]["max_eval_samples"])

    train_collator = Llama32SFTCollator(
        processor=processor,
        max_length=sp.max_length,
        training=True,
        add_generation_prompt=False,
    )

    batch = next(
        iter(
            DataLoader(
                train_datasets, batch_size=sp.batch_size, collate_fn=train_collator
            )
        )
    )
    print("supervised tokens per row:", (batch["labels"] != -100).sum(dim=1))
    assert (batch["labels"] != -100).any(), "All labels are -100 — no loss will flow."

    report_to = ["wandb"] if run is not None else []
    prefetch_factor = (
        extras.dataloader_prefetch_factor if sp.num_workers and sp.num_workers > 1 else None
    )
    sft_args = SFTConfig(
        output_dir=adapter_output_dir,
        num_train_epochs=sp.num_epochs,
        per_device_train_batch_size=sp.batch_size,
        per_device_eval_batch_size=sp.batch_size,
        gradient_accumulation_steps=sp.gradient_accumulation_steps,
        learning_rate=sp.lr,
        weight_decay=sp.weight_decay,
        warmup_steps=sp.warmup_steps,
        warmup_ratio=sp.warmup_ratio,
        max_grad_norm=sp.max_grad_norm,
        logging_steps=sp.logging_steps,
        save_steps=sp.save_steps,
        eval_steps=sp.eval_steps,
        save_total_limit=sp.save_total_limit,
        load_best_model_at_end=sp.load_best_model_at_end,
        metric_for_best_model=sp.metric_for_best_model,
        greater_is_better=sp.greater_is_better,
        fp16=sp.fp16,
        bf16=extras.bf16,
        tf32=extras.tf32,
        max_length=sp.max_length,
        eval_strategy=extras.evaluation_strategy,
        remove_unused_columns=False,
        report_to=report_to,
        logging_dir=logistics.sft_log_dir,
        save_safetensors=extras.save_safetensors,
        optim=extras.optim,
        dataloader_num_workers=sp.num_workers,
        dataloader_pin_memory=extras.dataloader_pin_memory,
        dataloader_persistent_workers=extras.dataloader_persistent_workers,
        dataloader_prefetch_factor=prefetch_factor,
        disable_tqdm=extras.disable_tqdm,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_datasets,
        eval_dataset=eval_dataset,
        data_collator=train_collator,
        args=sft_args,
    )
    trainer.add_callback(LogCallback(sp.logging_steps))

    last_checkpoint = get_last_checkpoint(adapter_output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.model.save_pretrained(adapter_output_dir, safe_serialization=True)
    processor.save_pretrained(adapter_output_dir)

    meta = {
        "base_model_name_or_path": model_name_or_path,
        "dataset_id": cfg["dataset"]["dataset_id"],
        "lang": cfg["dataset"]["lang"],
        "effective_batch_size": sp.batch_size * sp.gradient_accumulation_steps * _get_world_size(),
        "flash_attention_enabled": use_flash,
        "config_source": "config/logistics.py",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "train_size": _dataset_size(train_datasets),
        "eval_size": _dataset_size(eval_dataset),
        **_count_trainable_params(trainer.model),
    }
    with open(os.path.join(adapter_output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    _verify_adapter_artifacts(adapter_output_dir)
    log_run_metadata(run, meta)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
