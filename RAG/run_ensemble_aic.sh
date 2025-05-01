#!/bin/bash

PROJECT_DIR=~/LLMinds/RAG
VENV_DIR="$PROJECT_DIR/rag_env"
SELECTOR_MODEL=llama-7b   # options: llama-7b, qwen-1.5b, neo-small, distqwen-1.5b, mistral-7b, …

# Use the SAME dataset keys that were used when generating the individual model answers
DATASETS=("bio1" "bio2" "bio3" "nmt" "popular")
# DATASETS=("bio1")

echo "🔁 Activating environment …"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

for DATA in "${DATASETS[@]}"; do
  echo "🏆 Ensembling answers for $DATA with selector model $SELECTOR_MODEL …"

  python3 ensemble_rag.py \
    --data "$DATA" \
    --selector_model "$SELECTOR_MODEL" \
    --include_candidates \
    --outdir ensemble_responses
    # add "--include_candidates" above if you want raw answers in the output

  echo "Finished ensemble for $DATA"
  echo "---------------------------"
done

echo "🎉 All selected datasets ensembled!"
