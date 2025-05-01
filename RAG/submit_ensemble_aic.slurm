#!/bin/bash

#SBATCH --job-name=rag_ensemble
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1               # selector LLM benefits from a GPU; change to :0 if CPU is enough
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  # usually enough – adjust per model
#SBATCH --time=00:10:00
#SBATCH --output=out/%j_ensemble.out
#SBATCH --constraint=gpuram24G

echo "🏁 Ensemble job started at $(date) on $(hostname)"

# Go to project directory
cd ~/LLMinds/RAG

# Activate venv
source rag_env/bin/activate

# Run the ensemble script (will look for existing responses/* files)
./run_ensemble_aic.sh

echo "✅ Ensemble job finished at $(date)"
