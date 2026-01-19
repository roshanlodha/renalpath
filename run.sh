#!/bin/bash

# Pipeline Orchestration Script
# Usage: ./run_pipeline.sh

echo "=========================================="
echo "Starting Renal Tumor Classification Pipeline"
echo "=========================================="

# 1. Preprocessing
echo "[1/3] Preprocessing and Splitting..."
./env/bin/python main.py --mode preprocess
if [ $? -ne 0 ]; then
    echo "Preprocessing failed!"
    exit 1
fi

# 2. Training (ResNet Example)
echo "[2/5] Training ResNet50..."
./env/bin/python main.py --mode train --model_type resnet --epochs 5 --batch_size 32
if [ $? -ne 0 ]; then
    echo "ResNet Training failed!"
    exit 1
fi

# 3. Evaluation ResNet
echo "[3/5] Evaluating ResNet50..."
./env/bin/python main.py --mode evaluate --model_type resnet
if [ $? -ne 0 ]; then
    echo "ResNet Evaluation failed!"
    exit 1
fi

# 4. Training GSViT (If available)
echo "[4/5] Training GSViT..."
# Only run if base pickle exists
GSVIT_PKL="models/GSViT.pkl"
if [ -f "$GSVIT_PKL" ]; then
    ./env/bin/python main.py --mode train --model_type gsvit --epochs 5 --batch_size 32 --gsvit_path "$GSVIT_PKL"
    if [ $? -ne 0 ]; then
        echo "GSViT Training failed!"
        exit 1
    fi

    # 5. Evaluation GSViT
    echo "[5/5] Evaluating GSViT..."
    ./env/bin/python main.py --mode evaluate --model_type gsvit --gsvit_path "$GSVIT_PKL"
    if [ $? -ne 0 ]; then
        echo "GSViT Evaluation failed!"
        exit 1
    fi
else
    echo "GSViT pickle not found at $GSVIT_PKL. Skipping GSViT steps."
fi

echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "Check 'analysis/' folder for outputs."
echo "=========================================="
