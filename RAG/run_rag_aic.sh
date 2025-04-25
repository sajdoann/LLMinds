#!/bin/bash

# === CONFIG ===
PROJECT_DIR=~/LLMinds/RAG
VENV_DIR=$PROJECT_DIR/rag_env
MODEL_NAME=llama-7b #distqwen-1.5b  # options: llama-7b, neo-small, qwen
TOP_K=20

# === Select datasets to run ===
# Options: bio1, bio2, bio3, nmt, popular, def
DATASETS=("nmt")  # ← edit this list to your needs

# === Activate environment once ===
echo "🔁 Activating environment..."
source $VENV_DIR/bin/activate
cd $PROJECT_DIR

# === Loop over selected datasets ===
for DATA in "${DATASETS[@]}"; do

  case $DATA in
    "bio1")
      DOCUMENT_PATH='./devset/ukrbiology/book01/topic01-Різноманітність тварин/text.en.txt'
      QUESTIONS_PATH='./devset/ukrbiology/book01/topic01-Різноманітність тварин/questions.json'
      ;;
    "bio2")
      DOCUMENT_PATH='./devset/ukrbiology/book01/topic02-Процеси життєдіяльностітварин/text.en.txt'
      QUESTIONS_PATH='./devset/ukrbiology/book01/topic02-Процеси життєдіяльностітварин/questions.json'
      ;;
    "bio3")
      DOCUMENT_PATH= './devset/ukrbiology/book01/topic03-Поведінка\ тварин/text.en.txt'
      QUESTIONS_PATH='./devset/ukrbiology/book01/topic03-Поведінка\ тварин/questions.json'
      ;;
    "nmt")
      DOCUMENT_PATH="./devset/nmtclass/lecture01-eval/lecture01-eval_full.txt"
      QUESTIONS_PATH="./devset/nmtclass/lecture01-eval/my-nmt-questions.json"
      ;;
    "popular")
      DOCUMENT_PATH="./devset/popular/video-22/text.en.txt"
      QUESTIONS_PATH="./devset/popular/video-22/questions.json"
      ;;
    "def")
      DOCUMENT_PATH="documents.txt"
      QUESTIONS_PATH="questions.json"
      ;;
    *)
      echo "❌ Unknown dataset: $DATA"
      continue
      ;;
  esac

  echo "🚀 Running RAG system on $DATA with $MODEL_NAME..."
  python rag_system.py \
    --model $MODEL_NAME \
    --top_k $TOP_K \
    --document "$DOCUMENT_PATH" \
    --questions "$QUESTIONS_PATH" \
    --data "$DATA"

  echo "✅ Finished $DATA"
  echo "---------------------------"

done

echo "🎉 All selected datasets processed!"
