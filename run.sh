#!/bin/bash

# Pipeline Orchestration Script
# Usage: ./run.sh

echo "=========================================="
echo "Starting Renal Tumor Classification Pipeline"
echo "=========================================="

# --- Part 1: Multiclass Pipeline ---
echo ">>> Starting Multiclass Pipeline (5 Classes) <<<"

# 1. Preprocessing
echo "[1/7] Preprocessing and Splitting (Multiclass)..."
./env/bin/python main.py --mode preprocess
if [ $? -ne 0 ]; then
    echo "Preprocessing failed!"
    exit 1
fi

# 2. ResNet
echo "[2/7] Training & Evaluating ResNet50 (Multiclass)..."
./env/bin/python main.py --mode train --model_type resnet
if [ $? -ne 0 ]; then echo "ResNet Training failed!"; exit 1; fi
./env/bin/python main.py --mode evaluate --model_type resnet
if [ $? -ne 0 ]; then echo "ResNet Evaluation failed!"; exit 1; fi

# 3. ViT
echo "[3/7] Training & Evaluating ViT-B/16 (Multiclass)..."
./env/bin/python main.py --mode train --model_type vit
if [ $? -ne 0 ]; then echo "ViT Training failed!"; exit 1; fi
./env/bin/python main.py --mode evaluate --model_type vit
if [ $? -ne 0 ]; then echo "ViT Evaluation failed!"; exit 1; fi

# 4. GSViT
echo "[4/7] Training & Evaluating GSViT (Multiclass)..."
GSVIT_PKL="models/GSViT.pkl"
if [ -f "$GSVIT_PKL" ]; then
    ./env/bin/python main.py --mode train --model_type gsvit
    if [ $? -ne 0 ]; then echo "GSViT Training failed!"; exit 1; fi
    ./env/bin/python main.py --mode evaluate --model_type gsvit
    if [ $? -ne 0 ]; then echo "GSViT Evaluation failed!"; exit 1; fi
else
    echo "GSViT pickle not found at $GSVIT_PKL. Skipping GSViT steps."
fi


# --- Part 2: Binary Pipeline ---
echo ">>> Starting Binary Pipeline (RCC vs Other) <<<"

# 5. Binary Preprocessing
echo "[5/7] Preprocessing and Splitting (Binary)..."
./env/bin/python main.py --mode preprocess --binary
if [ $? -ne 0 ]; then
    echo "Binary Preprocessing failed!"
    exit 1
fi

# 6. Binary ResNet
echo "[6/7] Training & Evaluating ResNet50 (Binary)..."
./env/bin/python main.py --mode train --model_type resnet --binary
if [ $? -ne 0 ]; then echo "Binary ResNet Training failed!"; exit 1; fi
./env/bin/python main.py --mode evaluate --model_type resnet --binary
if [ $? -ne 0 ]; then echo "Binary ResNet Evaluation failed!"; exit 1; fi

# 7. Binary ViT
echo "[7/7] Training & Evaluating ViT-B/16 (Binary)..."
./env/bin/python main.py --mode train --model_type vit --binary
if [ $? -ne 0 ]; then echo "Binary ViT Training failed!"; exit 1; fi
./env/bin/python main.py --mode evaluate --model_type vit --binary
if [ $? -ne 0 ]; then echo "Binary ViT Evaluation failed!"; exit 1; fi

# 8. Binary GSViT
echo "[8/7] Training & Evaluating GSViT (Binary)..."
if [ -f "$GSVIT_PKL" ]; then
    ./env/bin/python main.py --mode train --model_type gsvit --binary
    if [ $? -ne 0 ]; then echo "Binary GSViT Training failed!"; exit 1; fi
    ./env/bin/python main.py --mode evaluate --model_type gsvit --binary
    if [ $? -ne 0 ]; then echo "Binary GSViT Evaluation failed!"; exit 1; fi
else
    echo "GSViT pickle not found at $GSVIT_PKL. Skipping GSViT steps."
fi

echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "Check 'analysis/' and 'models/' (including 'models/binary') for outputs."
echo "=========================================="