#!/bin/bash
# run_pipeline.sh - Execute full face recognition training pipeline
# Usage: ./run_pipeline.sh

set -e  # Exit on error

echo "=============================================="
echo "TinyML Face Recognition Pipeline"
echo "=============================================="

# Configuration
DATASET_DIR="dataset"
PROCESSED_DIR="processed"
MODEL_DIR="models"
TFLITE_DIR="tflite"
ARDUINO_DIR="G_arduino_firmware"

# Check for dataset
if [ ! -d "$DATASET_DIR" ]; then
    echo ""
    echo "ERROR: Dataset directory not found!"
    echo ""
    echo "Please create the following structure:"
    echo "  $DATASET_DIR/"
    echo "  ├── stage_a/"
    echo "  │   ├── person/      (500+ images)"
    echo "  │   └── no_person/   (500+ images)"
    echo "  ├── stage_b/"
    echo "  │   ├── person1/     (100+ images)"
    echo "  │   ├── person2/     (100+ images)"
    echo "  │   ├── person3/     (100+ images)"
    echo "  │   ├── person4/     (100+ images)"
    echo "  │   └── person5/     (100+ images)"
    echo "  └── unknown_test/"
    echo "      └── other_person*/ (50+ images each)"
    echo ""
    echo "See B_DATA_COLLECTION_PROTOCOL.md for details."
    exit 1
fi

# Step 1: Preprocess data
echo ""
echo "Step 1: Preprocessing and augmenting data..."
echo "----------------------------------------------"
python C_preprocess_and_augment.py \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$PROCESSED_DIR" \
    --augment_train \
    --augmentations 3 \
    --use_face_detection

# Step 2: Train models
echo ""
echo "Step 2: Training models..."
echo "----------------------------------------------"
python E_train_model.py \
    --data_dir "$PROCESSED_DIR" \
    --output_dir "$MODEL_DIR" \
    --epochs 100 \
    --batch_size 32

# Step 3: Quantize models
echo ""
echo "Step 3: Quantizing models to INT8..."
echo "----------------------------------------------"
python F_quantize_model.py \
    --model_dir "$MODEL_DIR" \
    --data_dir "$PROCESSED_DIR" \
    --output_dir "$TFLITE_DIR" \
    --validate

# Step 4: Copy to Arduino folder
echo ""
echo "Step 4: Copying model headers to Arduino folder..."
echo "----------------------------------------------"
cp "$TFLITE_DIR/stage_a_model.h" "$ARDUINO_DIR/"
cp "$TFLITE_DIR/stage_b_model.h" "$ARDUINO_DIR/"

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Generated files:"
echo "  - $MODEL_DIR/stage_a_final.keras"
echo "  - $MODEL_DIR/stage_b_final.keras"
echo "  - $TFLITE_DIR/stage_a_int8.tflite"
echo "  - $TFLITE_DIR/stage_b_int8.tflite"
echo "  - $ARDUINO_DIR/stage_a_model.h"
echo "  - $ARDUINO_DIR/stage_b_model.h"
echo ""
echo "Next steps:"
echo "  1. Open G_arduino_firmware/G_arduino_firmware.ino in Arduino IDE"
echo "  2. Install libraries: ArduCAM, Arduino_TensorFlowLite"
echo "  3. Select board: Arduino Nano 33 BLE"
echo "  4. Upload firmware"
echo "  5. Test with H_TESTING_PROTOCOL.md"
echo ""
