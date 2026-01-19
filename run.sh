#!/bin/bash

# Pipeline Orchestration Script
# Usage: ./run_pipeline.sh

echo "=========================================="
echo "Starting Renal Tumor Classification Pipeline"
echo "=========================================="

# 1. Preprocessing
echo "[1/3] Preprocessing and Splitting..."
python main.py --mode preprocess
if [ $? -ne 0 ]; then
    echo "Preprocessing failed!"
    exit 1
fi

# 2. Training (ResNet Example)
echo "[2/3] Training ResNet50..."
# Adjust epochs as needed. Using 5 for demonstration/quick run.
python main.py --mode train --model_type resnet --epochs 5 --batch_size 32
if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

# 3. Evaluation
echo "[3/3] Evaluating ResNet50..."
python main.py --mode evaluate --model_type resnet
if [ $? -ne 0 ]; then
    echo "Evaluation failed!"
    exit 1
fi

echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "=========================================="
