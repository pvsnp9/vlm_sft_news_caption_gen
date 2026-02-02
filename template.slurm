#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=AYASFT
#SBATCH --output=outputs/%u/aya/aya_sft.out
#SBATCH --error=outputs/%u/aya/aya_sft.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem-per-cpu=20GB
#SBATCH --export=ALL
#SBATCH --time=5-0:00:00

# Method A: use Slurm’s export directive to override any inherited HF_HOME
#SBATCH --export=ALL,HF_HOME=/dir/to/HF_CACHE

#useful commands for debugging
#sinfo -o "%P %l" - prints partition and time limit

set echo
umask 0027

# to see ID and state of GPUs assigned
# nvidia-smi

export HF_TOKEN=<your_token>
export WANDB_API_KEY=<your_key>

# module load gnu10                           
# module load python/3.9.9-jh
module purge

# Load necessary modules and set environment variables
module load cuda/12.2.0
export CUDA_HOME=/opt/sw/other/apps/cuda/12.2.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "FlashAttention-2 expects nvcc at: $CUDA_HOME/bin/nvcc"

echo "HF_HOME is now set to: $HF_HOME"

# export PYTHONPATH=/projects/mzampier/python-packages/projectX:$PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate <your_env> || { echo "Failed to activate the conda env"; exit 1; }

python - <<'PY'
try:
    import flash_attn 
    print("FlashAttention-2: available")
except Exception as exc:
    print(f"FlashAttention-2: unavailable ({exc})")
PY

python -m src.train.ayavisionsft || { echo "SFT script failed"; exit 1; }
