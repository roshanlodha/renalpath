# Tumor Classification Pipeline Walkthrough

**<span style="color:red;">Note: The code in the `legacy` subdirectory was used for the generation of the manuscript data.</span>** Code in the root directory is represents a refactored version of the code in the `legacy` subdirectory, developed for simplified pipeline usage/recreation. 

## Overview

The pipeline consists of the following components:
- **Data Preprocessing**: Filtering, resizing, and splitting (80/10/10).
- **Dataset**: Custom PyTorch Dataset with dynamic transformations.
- **Models**:
    - **ResNet-50**: Pretrained on ImageNet, fine-tuned.
    - **GSViT**: Pretrained backbone, custom MLP head.
- **Training**: Weighted CrossEntropyLoss, Adam optimizer, StepLR, Early Stopping.
- **Evaluation**: Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix.

## Files

- `preprocess.py`: Static preprocessing and splitting.
- `dataset.py`: `TumorDataset` class and transforms.
- `resnet_model.py`: ResNet-50 definition.
- `gsvit_model.py`: GSViT wrapper.
- `train.py`: Training loop.
- `evaluate.py`: Evaluation metrics.
- `main.py`: Entry point.

## Usage

### 1. Preprocessing
Run this once to prepare the data.
```bash
python main.py --mode preprocess --data_dir /path/to/raw --processed_dir /path/to/processed --metadata_csv /path/to/meta.csv
```

### 2. Training
Train ResNet-50:
```bash
python main.py --mode train --model_type resnet --processed_dir /path/to/processed --epochs 30
```

Train GSViT:
```bash
python main.py --mode train --model_type gsvit --gsvit_path GSViT.pkl --processed_dir /path/to/processed --epochs 30
```

### 3. Evaluation
Evaluate the best model:
```bash
python main.py --mode evaluate --model_type resnet --processed_dir /path/to/processed
```

### 4. Verification (Dry Run)
Check if models load correctly without data:
```bash
python main.py --mode dry_run --gsvit_path GSViT.pkl
```

## Requirements
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- opencv-python
- matplotlib
- seaborn
- tqdm