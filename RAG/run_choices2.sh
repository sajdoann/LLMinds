#!/bin/bash

ROOT_DIR="testset"
TOP_K=20
MODEL_NAME=llama-7b #distqwen-1.5b
QUESTIONS_COUNT=0
CTR=0
jq -r '.[][]' choices2.json | while IFS= read -r RAW_DIR; do
    DIR="testset/$RAW_DIR"
    CTR=$(($CTR+1))
   # echo "ctr $CTR"

    if [[ ! -d "$DIR" ]]; then
        echo "Directory not found: $DIR"
        continue
    fi
        # Find all *_PRIMARY.json files within up to 2 levels of the directory
    mapfile -t FOUND_FILES < <(find "$DIR" -maxdepth 2 -type f -name '*PRIMARY.json')

    if [[ ${#FOUND_FILES[@]} -eq 0 ]]; then
        echo "ERROR: skipping, no *PRIMARY.json in $DIR"
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

    for QUESTION_FILE in "${FOUND_FILES[@]}"; do
        if [[ ! -f "$QUESTION_FILE" ]]; then
            echo "  Skipping non-existent file: $QUESTION_FILE"
            continue
        fi
       echo "q: $QUESTION_FILE"
       CUR_QUESTION_COUNT=$(jq length "$QUESTION_FILE")
       
       FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
	if (( FREE_MEM < 500 )); then
    		echo "Not enough GPU memory ($FREE_MEM MiB). Sleeping."
    		pkill -f rag_system.py
		sleep 30
	fi

       echo "CUR_QUESTION_COUNT: $CUR_QUESTION_COUNT"
       QUESTIONS_COUNT=$((QUESTIONS_COUNT + CUR_QUESTION_COUNT))
       
       echo "so far generated: $QUESTIONS_COUNT"
	 python3 rag_system.py \
         --model $MODEL_NAME \
         --top_k $TOP_K \
         --document "$TEXT_FILE" \
         --questions "$QUESTION_FILE"
	done

done


