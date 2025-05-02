#!/bin/bash

ROOT_DIR="testset"
TOP_K=20
MODEL_NAME=distqwen-1.5b
QUESTIONS_COUNT=0

while read -r DIR; do
    if [[ -f "$DIR/questions_by_GOLD_PRIMARY.json" ]]; then
        QUESTION_FILE="$DIR/questions_by_GOLD_PRIMARY.json"
    elif [[ -f "$DIR/questions_by_GPT_PRIMARY.json" ]]; then
        QUESTION_FILE="$DIR/questions_by_GPT_PRIMARY.json"
    else
        echo "ERROR: skipping, no gold or gpt in $DIR"
        continue
    fi

    if [[ -f "$DIR/text.en.txt" ]]; then
        TEXT_FILE="$DIR/text.en.txt"
    elif [[ -f "$DIR/text.txt" ]]; then
        TEXT_FILE="$DIR/text.txt"
    else
        echo "No text file found in $DIR. Skipping."
        continue
    fi

    echo "🚀 Starting RAG system with $MODEL_NAME on $QUESTION_FILE..."

    CUR_QUESTION_COUNT=$(jq length "$QUESTION_FILE")
    QUESTIONS_COUNT=$((QUESTIONS_COUNT + CUR_QUESTION_COUNT))
    echo "CUR_QUESTION_COUNT: $CUR_QUESTION_COUNT"
    echo "so far generated: $QUESTIONS_COUNT"
    #python3 rag_system.py \
    #     --model $MODEL_NAME \
    #     --top_k $TOP_K \
    #     --document "$TEXT_FILE" \
    #     --questions "$QUESTION_FILE"
done < <(find "$ROOT_DIR" -type d)

echo "Total questions: $QUESTIONS_COUNT"

