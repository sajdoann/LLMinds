#!/bin/bash

# === CONFIG ===
PROJECT_DIR=~/LLMinds/questionsGen/
VENV_DIR=~/LLMinds/RAG/rag_env
MODEL_NAME=distqwen-1.5b  # options: llama-7b, neo-small, qwen





# === Activate environment once ===
echo "🔁 Activating environment..."
source $VENV_DIR/bin/activate
cd $PROJECT_DIR

python3 example.py


echo "🎉 All selected datasets processed!"
