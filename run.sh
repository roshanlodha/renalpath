#!/bin/bash

# Pipeline Orchestration Script
# Usage: ./run_pipeline.sh

echo "=========================================="
echo "Starting Renal Tumor Classification Pipeline"
echo "=========================================="

# 1. Preprocessing
echo "[1/7] Preprocessing and Splitting..."
./env/bin/python main.py --mode preprocess
if [ $? -ne 0 ]; then
    echo "Preprocessing failed!"
    exit 1
fi

# 2. Training (ResNet Example)
echo "[2/7] Training ResNet50..."
./env/bin/python main.py --mode train --model_type resnet --epochs 100 --batch_size 32
if [ $? -ne 0 ]; then
    echo "ResNet Training failed!"
    exit 1
fi

# 3. Evaluation ResNet
echo "[3/7] Evaluating ResNet50..."
./env/bin/python main.py --mode evaluate --model_type resnet
if [ $? -ne 0 ]; then
    echo "ResNet Evaluation failed!"
    exit 1
fi

# 4. Training ViT
echo "[4/7] Training ViT-B/16..."
./env/bin/python main.py --mode train --model_type vit --epochs 100 --batch_size 32
if [ $? -ne 0 ]; then
    echo "ViT Training failed!"
    exit 1
fi

# 5. Evaluation ViT
echo "[5/7] Evaluating ViT-B/16..."
./env/bin/python main.py --mode evaluate --model_type vit
if [ $? -ne 0 ]; then
    echo "ViT Evaluation failed!"
    exit 1
fi

# 6. Training GSViT (If available)
echo "[6/7] Training GSViT..."
# Only run if base pickle exists
GSVIT_PKL="models/GSViT.pkl"
if [ -f "$GSVIT_PKL" ]; then
    ./env/bin/python main.py --mode train --model_type gsvit --epochs 100 --batch_size 32 --gsvit_path "$GSVIT_PKL"
    if [ $? -ne 0 ]; then
        echo "GSViT Training failed!"
        exit 1
    fi

    # 7. Evaluation GSViT
    echo "[7/7] Evaluating GSViT..."
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
