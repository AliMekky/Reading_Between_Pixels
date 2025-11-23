#!/bin/bash

# Script to run inference across different models and image variants
# Usage: bash run_inference.sh

# Define models
MODELS=(
    "qwen-vl"
    "llava-next"
    "internvl"
    "llava"
)

# Define image variants
VARIANTS=(
    "correct"
    "relevant"
    "irrelevant"
    "misleading"
    "notext"
)

# Base paths
IMAGE_BASE="./filtered_data"
QUESTIONS_BASE="./filtered_data"
OUTPUT_BASE="./test_scripts"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_BASE"

# Log file
LOG_FILE="$OUTPUT_BASE/inference_log.txt"
echo "Starting inference runs at $(date)" | tee "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"

# Loop through each model
for model in "${MODELS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "Processing model: $model" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    
    # Loop through each variant
    for variant in "${VARIANTS[@]}"; do
        echo "" | tee -a "$LOG_FILE"
        echo "  Running variant: $variant" | tee -a "$LOG_FILE"
        
        # Define paths
        IMAGE_FOLDER="$IMAGE_BASE/$variant"
        QUESTIONS_FILE="$QUESTIONS_BASE/filtered_questions_${variant}.json"
        OUTPUT_FILE="$OUTPUT_BASE/${model}_${variant}_results.json"
        
        # Check if input files exist
        if [ ! -d "$IMAGE_FOLDER" ]; then
            echo "  ⚠️  Warning: Image folder not found: $IMAGE_FOLDER" | tee -a "$LOG_FILE"
            continue
        fi
        
        if [ ! -f "$QUESTIONS_FILE" ]; then
            echo "  ⚠️  Warning: Questions file not found: $QUESTIONS_FILE" | tee -a "$LOG_FILE"
            continue
        fi
        
        # Run inference
        echo "  Command: python infere_vlms.py --model_type $model --image_folder $IMAGE_FOLDER --questions $QUESTIONS_FILE --output $OUTPUT_FILE" | tee -a "$LOG_FILE"
        
        START_TIME=$(date +%s)
        
        python infere_vlms.py \
            --model_type "$model" \
            --image_folder "$IMAGE_FOLDER" \
            --questions "$QUESTIONS_FILE" \
            --output "$OUTPUT_FILE" 2>&1 | tee -a "$LOG_FILE"
        
        EXIT_CODE=${PIPESTATUS[0]}
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "  ✅ Success! Duration: ${DURATION}s" | tee -a "$LOG_FILE"
            echo "  Output saved to: $OUTPUT_FILE" | tee -a "$LOG_FILE"
        else
            echo "  ❌ Failed with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
        fi
        
    done
done

echo "" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "All inference runs completed at $(date)" | tee -a "$LOG_FILE"
echo "Results saved in: $OUTPUT_BASE" | tee -a "$LOG_FILE"
echo "Log saved in: $LOG_FILE" | tee -a "$LOG_FILE"