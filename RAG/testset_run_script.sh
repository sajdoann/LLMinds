#!/bin/bash

# Set the root directory and the Python script to call
ROOT_DIR="testset"
PYTHON_SCRIPT="your_script.py"  # Change this to your actual script
TOP_K=15
MODEL_NAME=distqwen-1.5b #llama-7b

# Recursively find directories
find "$ROOT_DIR" -type d | while read -r DIR; do
    # Find the first *_PRIMARY.json file in the directory
    QUESTION_FILE=$(find "$DIR" -maxdepth 1 -type f -name "*_PRIMARY.json" | head -n 1)

    if [[ -n "$QUESTION_FILE" ]]; then
        # Check for preferred text files
        if [[ -f "$DIR/text.en.txt" ]]; then
            TEXT_FILE="$DIR/text.en.txt"
        elif [[ -f "$DIR/text.txt" ]]; then
            TEXT_FILE="$DIR/text.txt"
        else
            echo "No text file found in $DIR. Skipping."
            continue
        fi

        echo "Running $PYTHON_SCRIPT with:"
        echo "  Question file: $QUESTION_FILE"
        echo "  Text file:     $TEXT_FILE"

        #python "$PYTHON_SCRIPT" "$QUESTION_FILE" "$TEXT_FILE"
        echo "🚀 Running RAG system on $DATA with $MODEL_NAME..."
        python rag_system.py \
          --model $MODEL_NAME \
          --top_k $TOP_K \
          --document "$DOCUMENT_PATH" \
          --questions "$QUESTIONS_PATH" \
          --data "$DOCUMENT_PATH"
        break
    fi
done
