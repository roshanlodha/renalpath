# Renal Tumor Classification Pipeline: LITE Methods Paper

## 1. Project Overview
This project implements a deep learning pipeline for automating the classification of renal tumors into distinct subtypes (the number of classes is inferred from `data/processed/classes.npy`, with a fallback default of 5). It supports **ResNet50** (Transfer Learning), a baseline **ViT-B/16**, and **GSViT** (Global-Local Transformer). The pipeline is designed to be robust, handling class imbalance and varying image qualities through advanced preprocessing and loss functions.

## 2. Methodology

### A. Preprocessing & Data Ingestion
The preprocessing pipeline (`preprocess.py`, `dataset.py`) ensures high-quality input for the models:

*   **Quality Control**: Images with mean pixel intensity < 20 and standard deviation < 10 are strictly filtered out (regarded as "black" or non-informative images).
*   **Border Removal**: A dynamic cropping mechanism uses Otsu's thresholding to identify the tissue region. The largest contour is found, and the image is cropped to its bounding box to remove black borders.
*   **Resizing & Padding**:
    *   Images smaller than `1024x1024` are padded to `1024x1024` before processing.
    *   A center crop of `1024x1024` is applied.
    *   Final resizing ensures all inputs are `(224, 224)` for model compatibility.
*   **Data Splitting**: Patient-level stratified split (by `Class`) grouped by `Patient ID` to prevent leakage:
    *   **Train+Val**: 50% of patients
    *   **Test**: 50% of patients
    *   **Val**: 20% of Train+Val (i.e., 10% overall), leaving 40% overall for Train
    *   Split configuration is recorded in `data/processed/split_config.json`.

### B. Input Normalization & Augmentation
*   **Augmentation (Train Only)**:
    *   `RandomHorizontalFlip`
    *   `RandomVerticalFlip`
    *   `ColorJitter` (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
*   **Normalization**:
    *   **ResNet50**: Standard ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Input format: RGB.
    *   **ViT-B/16**: Standard ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Input format: RGB.
    *   **GSViT**: Zero-centered normalization (`mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`). Input format: BGR (channels flipped).

### C. Model Architectures
#### 1. ResNet50 (Baseline)
*   **Backbone**: ResNet50 pretrained on ImageNet-1K V2.
*   **Classification Head**: The original fully connected layer is replaced with a custom blocks:
    1.  `Linear(2048 -> 512)`
    2.  `BatchNorm1d(512)`
    3.  `ELU` Activation
    4.  `Dropout(p=0.3)`
    5.  `Linear(512 -> Num_Classes)`

### D. Analysis & Interpretation
All figures are saved under `analysis/{model}/`:
1.  **Confusion Matrix** (`analysis/{model}/confusion.png`): Visualizes misalignment between predicted and true labels.
2.  **AUPRC Bar Plot** (`analysis/{model}/auprc.png`): AUPRC per class, with dashed-outline bars showing **training-set class prevalence**.
3.  **Feature Correlation Heatmap** (`analysis/{model}/feature_corr.png`): (GSViT/ViT) Feature–histology correlation with a fixed color scale from **-1 to 1**.
4.  **Training Curves** (`analysis/{model}/training_curves.png`): Train/val loss and accuracy over epochs (saved during training).

## 4. Repository Structure & Usage

### Key Scripts
*   **`run.sh`**: The master orchestration script. Runs: Preprocessing -> ResNet Train/Eval -> ViT Train/Eval -> GSViT Train/Eval (if weights exist).
*   **`main.py`**: The central entry point. Handles argument parsing and dispatches tasks to other modules.
    *   `python main.py --mode preprocess`: Runs data cleaning and splitting.
    *   `python main.py --mode train --model_type [resnet|vit|gsvit]`: Trains the specified model (default `--epochs 10`).
    *   `python main.py --mode evaluate --model_type [resnet|vit|gsvit]`: Evaluates the best saved model on the test set.
*   **`dataset.py`**: Defines the `TumorDataset` class, managing image loading, border removal, and transforms.
*   **`train.py`**: Contains the training loop, including metric tracking, checkpointing, and sampler logic.
*   **`models.py`**: Defines the classes `ResNet50_Classifier`, `ViT_Classifier`, and `GSViT_Classifier`.
*   **`evaluate.py`**: Handles metric calculation (Acc, F1, AUPRC) and plotting.

### How to Run
1.  **Prepare Environment**: Ensure dependencies in `requirements.txt` are installed.
2.  **Execute Pipeline**:
    ```bash
    ./run.sh
    ```
    Outputs (models) will be in `models/`.
    Plots and analysis figures will be in `analysis/{model}/`.
