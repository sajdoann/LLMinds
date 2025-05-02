#!/bin/bash
#SBATCH --job-name=rag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G #32 sometimes enough
#SBATCH --time=50:00:00
#SBATCH --output=out/%j.out

echo "🏁 Job started at $(date) on $(hostname)"


# Go to working directory
cd ~/LLMinds/RAG

# Activate venv
source rag_env/bin/activate

# Run script
./run_choices2.sh

echo "✅ Job finished at $(date)"

