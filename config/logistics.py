import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Logistics:
    output_root_dir: str = "outputs"
    models_output_dir: str = "outputs/models"
    data_dir: str = "data"
    processed_data_dir: str = "data/processed"
    hf_cache_dir: str = "data/hf_cache"
    sft_log_dir: str = "outputs/logs/sft"
    results_dir:str = "outputs/results"
    reports_dir: str = "outputs/reports"


    hf_datatset_id:str = "tharindu/XL-MUNIChus"

    splits: List[str] = field(
        default_factory=lambda: ["train", "validation", "test"]
    )

    langs: List[str] = field(default_factory=lambda: ["en", "zh"])
    infer_bathc_size:int = 4
    gen_max_token:int = 128


    hf_token:any = None
    wandb_token: any = None
    wandb_project: str = "news_caption_generation_sft"
    wandb_tags: List[str] = field(default_factory=lambda: ["news_caption_gen", "sft"])



#------------------------------
# Model Cards
# ------------------------------

@dataclass(frozen=True)
class ModelCards:
    llama3_2vl: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    aya_model_name: str = "CohereLabs/aya-vision-8b"
    qwen3_vl_8b_instruct: str = "Qwen/Qwen3-VL-8B-Instruct"

    # default training sequence length
    max_length: int = 2048

    llama_max_length:int = 2048
    aya_max_length: int = 2048
    qwen3_max_length:int = 2048



# ------------------------------
# QLoRA (bitsandbytes) Params
# ------------------------------
@dataclass(frozen=True)
class QLoRAParams:
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

    # Compute dtype for 4-bit matmuls
    # On A100: bf16 is ideal
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Optional extras (handy to keep)
    llm_int8_threshold: float = 6.0   # unused for pure 4-bit, safe to keep
    llm_int8_has_fp16_weight: bool = False


# ------------------------------
# LoRA Params
# ------------------------------
@dataclass(frozen=True)
class LoRAParams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"

    # attention + MLP projections
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


# ------------------------------
# SFT Parameters (A100 optimized)
# ------------------------------
@dataclass
class SFTParams:
    # Training length
    num_epochs: int = 3                   # start with 1; go to 2 if still improving

    # Dataloader
    num_workers: int = 10                 # 8–12 usually best on A100 nodes

    # Optimization
    lr: float = 1e-4                      # LoRA SFT standard (2e-5 is typically too low)
    weight_decay: float = 0.0             # adapters generally don’t need wd
    warmup_steps: int = 0                 # use warmup_ratio instead
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # Batch sizing
    batch_size: int = 4                   # per-device micro-batch; drop to 2 if images are large
    gradient_accumulation_steps: int = 8  # effective batch = 32 (good default)

    # Sequence
    max_length: int = 2048


    # Precision (prefer bf16 in  SFTConfig; keep fp16 off)
    fp16: bool = False

    # Logging / checkpointing
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 2

    # Model selection
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Repro
    seed: int = 42

    # Output
    model_dir: str = "/projects/mzampier/tsuyog/vlm_sft_news_caption_gen/outputs/models"

    # Early stopping (only if you actually add the callback)
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01

    # Inference / eval decode (make deterministic for classification)
    do_sample: bool = False
    top_p: float = 1.0
    temperature: float = 0.0


# ------------------------------
# (Optional) SFTConfig - extras you should pass explicitly
# ------------------------------
@dataclass
class SFTConfigExtras:
    # Speed / memory
    bf16: bool = True                     # A100 supports bf16
    tf32: bool = True
    gradient_checkpointing: bool = True
    use_cache: bool = False               # must be False when checkpointing

    # Attention acceleration
    use_flash_attention_2: bool = True    # try; fallback gracefully

    # Optimizer
    optim: str = "paged_adamw_8bit"

    # Dataloader tuning
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2

    # Trainer display
    disable_tqdm: bool = False

    # Saving
    save_safetensors: bool = True

    # Eval strategy
    evaluation_strategy: str = "steps"


def build_cfg(model_name_or_path: str) -> dict:
    logistics = Logistics()
    logistics.wandb_tags.append(model_name_or_path)
    safe_model_name = model_name_or_path.replace("/", "-")
    run_name = f"{safe_model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return {
        "logistics": logistics,
        "qlora": QLoRAParams(),
        "lora": LoRAParams(),
        "sft": SFTParams(),
        "sft_extras": SFTConfigExtras(),
        "model": {
            "base_model_name_or_path": model_name_or_path,
            "use_flash_attention": True,
            "torch_dtype": "bfloat16",
            "gradient_checkpointing": True,
            "use_cache": False,
        },
        "dataset": {
            "dataset_id": logistics.hf_datatset_id,
            "lang": "en",
            "train_split": "train",
            "eval_split": "validation",
            "fallback_eval_split": "test",
            "streaming": False,
            "max_train_samples": None,
            "max_eval_samples": 8,
        },
        "output": {
            "adapter_output_dir": None,
            "model_dir": safe_model_name,
            "run_name": run_name,
        },
        "wandb": {
            "project": os.environ.get("WANDB_PROJECT", logistics.wandb_project),
            "entity": None,
            "tags": logistics.wandb_tags,
            "log_model": False,
        },
        "collator": {
            "image_key": "image",
            "query_key": "query",
            "caption_key": "caption",
            "target_key": "caption",
            "system_prompt": None,
        },
        "eval_decode": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 8,
        },
    }


PROMPT_TMPL = (
    "You are writing a caption for a newspaper image.\n"
    "Given the image and this news article excerpt:\n"
    "{news}\n"
    "Task: Write a concise, informative caption for this image in {language}.\n"
    "Guidelines:\n"
    "- Write in {language} language only\n"
    "- Keep it brief\n"
    "- Identify and include: people's names, locations, and organisations\n"
    "- Connect what you see in the image to the news context\n"
    "- Use journalistic style (factual, clear, objective)\n"
    "- Focus on the main subject of the image\n"
    "Caption in {language}:"
)

SYS_PROMPT = (
    "Be factual. Do not invent identities or details. Follow requested language and format exactly."
)

PROMPT_TPML_V1 = (
    "You are a newspaper photo editor.\n"
    "NEWS EXCERPT:\n{news}\n"
    "LANGUAGE: {language}\n\n"
    "Write ONE publication-ready caption in {language} (1-2 sentences, ~35 words max).\n"
    "Rules: be strictly factual; don't guess. Include names/locations/organizations only if in the excerpt or clearly readable in the image.\n"
    "Start with the main visible subject; link it to the excerpt; avoid intent/causality claims.\n\n"
    "Caption ({language}):"
)
