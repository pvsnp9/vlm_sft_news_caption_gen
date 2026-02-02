# VLM SFT for News Caption Genration 

Fine-tuning scripts for vision-language models (VLMs) with Slurm-based training jobs.

**Work in progress:** This repository supports an active research paper project and is under ongoing development.

## Repository layout
- `src/train/`: training entrypoints
- `slurm/`: Slurm job configurations that launch the training scripts
- `config/`, `src/dataset/`, `src/collators/`: supporting code
- `outputs/`: training logs and checkpoints (by default)

## Requirements
- Python environment with dependencies from `requirements.txt`
- Slurm + CUDA-capable GPU nodes
- Access to the target model weights and datasets
- Environment variables for external services (e.g., Hugging Face, W&B)

## Setup
1) Create/activate your Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

2) Set required environment variables (do not hardcode secrets in repo files):

```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
```

3) Review the Slurm scripts under `slurm/` and update (use `template.slurm` as the reference for Slurm and environment setup):
- `#SBATCH` resources (partition, GPUs, memory, time)
- `HF_HOME` (if you want a custom cache location)
- `conda activate` path (or replace with your preferred environment setup)

## Run (Slurm)
Each training entrypoint in `src/train/` has a corresponding Slurm script.
Submit a job with `sbatch`:

```bash
sbatch slurm/qwen3sft.slurm
sbatch slurm/llamasft.slurm
sbatch slurm/ayaaft.slurm
```

## Outputs
- Logs: `outputs/<user>/<model>/<name>.out` and `.err`
- Checkpoints: under `outputs/models/` (if enabled by the training script)
