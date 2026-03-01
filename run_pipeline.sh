#!/bin/bash

# TinyML Person Detection - Complete Pipeline Execution Script
# 
# This script runs the entire pipeline from data preprocessing to model conversion.
# Assumes you have already collected your dataset in the 'dataset/' directory.
#
# Usage: bash run_pipeline.sh

set -e  # Exit on error

echo "========================================"
echo "TinyML Person Detection Pipeline"
echo "========================================"
echo ""

# Configuration
DATASET_DIR="dataset"
PREPROCESSED_DIR="preprocessed_dataset"
TRAINING_OUTPUT="training_output"
TFLITE_OUTPUT="tflite_models"
IMAGE_SIZE=96
MODEL_SIZE="auto"
BATCH_SIZE=32
EPOCHS=50

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory '$DATASET_DIR' not found!"
    echo "Please create it with the following structure:"
    echo "  $DATASET_DIR/"
    echo "    person/       # Images with people"
    echo "    no_person/    # Images without people"
    exit 1
fi

# Check if dataset has required subdirectories
if [ ! -d "$DATASET_DIR/person" ] || [ ! -d "$DATASET_DIR/no_person" ]; then
    echo "ERROR: Dataset must contain 'person/' and 'no_person/' subdirectories"
    exit 1
fi

# Count images
PERSON_COUNT=$(find "$DATASET_DIR/person" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
NO_PERSON_COUNT=$(find "$DATASET_DIR/no_person" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)

echo "Dataset statistics:"
echo "  Person images:    $PERSON_COUNT"
echo "  No-person images: $NO_PERSON_COUNT"
echo "  Total images:     $((PERSON_COUNT + NO_PERSON_COUNT))"
echo ""

if [ $PERSON_COUNT -lt 100 ] || [ $NO_PERSON_COUNT -lt 100 ]; then
    echo "WARNING: Very small dataset (<100 images per class)"
    echo "Recommendation: Collect at least 1,000 images per class"
    echo ""
fi

# Step 1: Preprocess dataset
echo "========================================"
echo "STEP 1/4: Preprocessing Dataset"
echo "========================================"
echo "Configuration:"
echo "  Input directory:  $DATASET_DIR"
echo "  Output directory: $PREPROCESSED_DIR"
echo "  Image size:       ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo ""

python preprocess_dataset.py \
    --input_dir "$DATASET_DIR" \
    --output_dir "$PREPROCESSED_DIR" \
    --size $IMAGE_SIZE \
    --verify

if [ $? -ne 0 ]; then
    echo "ERROR: Preprocessing failed!"
    exit 1
fi

echo ""
echo "✓ Preprocessing complete!"
echo ""

# Step 2: Train model
echo "========================================"
echo "STEP 2/4: Training Model"
echo "========================================"
echo "Configuration:"
echo "  Dataset:    $PREPROCESSED_DIR"
echo "  Model size: $MODEL_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs:     $EPOCHS"
echo ""

python train_model.py \
    --dataset_dir "$PREPROCESSED_DIR" \
    --model_size "$MODEL_SIZE" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --output_dir "$TRAINING_OUTPUT"

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo ""
echo "✓ Training complete!"
echo ""

# Step 3: Convert to TensorFlow Lite
echo "========================================"
echo "STEP 3/4: Converting to TensorFlow Lite"
echo "========================================"
echo "Configuration:"
echo "  Model:       $TRAINING_OUTPUT/best_model.h5"
echo "  Dataset:     $PREPROCESSED_DIR"
echo "  Output dir:  $TFLITE_OUTPUT"
echo ""

python convert_to_tflite.py \
    --model "$TRAINING_OUTPUT/best_model.h5" \
    --dataset_dir "$PREPROCESSED_DIR" \
    --output_dir "$TFLITE_OUTPUT"

if [ $? -ne 0 ]; then
    echo "ERROR: TFLite conversion failed!"
    exit 1
fi

echo ""
echo "✓ TFLite conversion complete!"
echo ""

# Step 4: Prepare Arduino deployment
echo "========================================"
echo "STEP 4/4: Preparing Arduino Deployment"
echo "========================================"

# Copy model_data.h to Arduino sketch directory
if [ -f "$TFLITE_OUTPUT/model_data.h" ]; then
    cp "$TFLITE_OUTPUT/model_data.h" arduino_person_detection/
    echo "✓ Copied model_data.h to arduino_person_detection/"
else
    echo "ERROR: model_data.h not found in $TFLITE_OUTPUT/"
    exit 1
fi

echo ""

# Print summary
echo "========================================"
echo "PIPELINE COMPLETE!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  ✓ Preprocessed dataset: $PREPROCESSED_DIR/"
echo "  ✓ Trained model:        $TRAINING_OUTPUT/best_model.h5"
echo "  ✓ TFLite model:         $TFLITE_OUTPUT/person_detector_int8.tflite"
echo "  ✓ Arduino header:       arduino_person_detection/model_data.h"
echo ""
echo "Model statistics:"
if [ -f "$TRAINING_OUTPUT/training_summary.json" ]; then
    python -c "
import json
with open('$TRAINING_OUTPUT/training_summary.json', 'r') as f:
    data = json.load(f)
    print(f\"  Parameters:     {data['model_config']['total_parameters']:,}\")
    print(f\"  Train accuracy: {data['final_metrics']['train_accuracy']*100:.2f}%\")
    print(f\"  Val accuracy:   {data['final_metrics']['val_accuracy']*100:.2f}%\")
"
fi

if [ -f "$TFLITE_OUTPUT/conversion_summary.json" ]; then
    python -c "
import json
with open('$TFLITE_OUTPUT/conversion_summary.json', 'r') as f:
    data = json.load(f)
    print(f\"  Model size:     {data['int8_tflite_size_kb']:.2f} KB\")
    print(f\"  INT8 accuracy:  {data['int8_validation_accuracy']*100:.2f}%\")
"
fi

echo ""
echo "Next steps:"
echo "  1. Open Arduino IDE"
echo "  2. Open arduino_person_detection/arduino_person_detection.ino"
echo "  3. Connect Arduino Nano 33 Sense"
echo "  4. Install required libraries (see ARDUINO_DEPLOYMENT_GUIDE.md)"
echo "  5. Upload firmware"
echo "  6. Open Serial Monitor (115200 baud)"
echo ""
echo "For detailed Arduino setup, see ARDUINO_DEPLOYMENT_GUIDE.md"
echo ""
echo "🚀 Happy deploying!"
