import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix, 
    matthews_corrcoef, 
    balanced_accuracy_score,
    f1_score
)
from sklearn.preprocessing import label_binarize
import os

def evaluate_model(
    model,
    dataloader,
    device,
    num_classes=5,
    class_names=None,
    output_dir='analysis',
    model_name='model',
    train_class_prevalence=None,
):
    """
    Evaluates the model and generates specific plots:
    1. Confusion Matrix (confusion.png)
    2. AUPRC Bar Plot (auprc.png)
    3. Feature Correlation Heatmap (feature_corr.png) [GSViT/ViT]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    
    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Check if model supports return_features
            # We implemented return_features in both ResNet and GSViT now
            try:
                logits, features = model(inputs, return_features=True)
            except TypeError:
                # Fallback if somehow using old model code (unlikely)
                logits = model(inputs)
                features = None
            except RuntimeError as e:
                msg = str(e).lower()
                if "cannot return features" in msg or "return features" in msg or "recognizable head" in msg:
                    logits = model(inputs)
                    features = None
                else:
                    raise
            
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if features is not None:
                all_features.extend(features.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    if len(all_features) > 0:
        all_features = np.array(all_features)
    
    # Ensure class names exist
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
        
    # --- Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    
    # --- Plot 1: Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix ({model_name})')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    fig.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion.png')
    fig.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Confusion Matrix to {cm_path}")
    
    # --- Plot 2: AUPRC Bar Plot ---
    from sklearn.metrics import average_precision_score
    
    y_test_bin = label_binarize(all_labels, classes=range(num_classes))
    auprc_scores = []
    
    # Handle single class case or missing classes if necessary, but assuming splits are good
    if y_test_bin.shape[1] == num_classes:
        for i in range(num_classes):
            ap = average_precision_score(y_test_bin[:, i], all_probs[:, i])
            auprc_scores.append(ap)
            print(f"Class {class_names[i]} AUPRC: {ap:.4f}")
            
        x = np.arange(num_classes)
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_classes))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, auprc_scores, color=colors, label='AUPRC')

        if train_class_prevalence is not None and len(train_class_prevalence) == num_classes:
            ax.bar(
                x,
                train_class_prevalence,
                fill=False,
                edgecolor='black',
                linewidth=2,
                linestyle='--',
                label='Train prevalence',
            )

        ax.set_title(f'AUPRC per Class ({model_name})')
        ax.set_ylabel('Average Precision Score')
        ax.set_xlabel('Cell Type')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x, class_names, rotation=45, ha='right')
        ax.legend()

        auprc_path = os.path.join(output_dir, 'auprc.png')
        fig.tight_layout()
        fig.savefig(auprc_path, dpi=200)
        plt.close(fig)
        print(f"Saved AUPRC Plot to {auprc_path}")
    else:
        print("Skipping AUPRC due to class mismatch in binarization.")

    # --- Plot 3: Feature Correlation Heatmap (GSViT/ViT) ---
    if model_name in {'gsvit', 'vit'} and len(all_features) > 0:
        # X: Features (1000s), Y: Cell Types
        # We need correlation between each feature and each class (one-hot)
        # Dimensions: Features (D) x Classes (C)
        
        # Normalize features? Pearson Correlation is scale invariant relative to itself, but centering helps.
        # np.corrcoef expects rows=variables, cols=observations.
        
        # We want corr(Feature_j, Class_k)
        # Construct matrix of [Features | Classes] -> Corr Matrix
        
        # D = all_features.shape[1]
        # C = num_classes
        
        # Calculating full correlation matrix (D+C)x(D+C) is huge if D is large.
        # Check size. ResNet=2048, GSViT usually 768 or 1024. It's manageable.
        
        # One-hot labels
        y_onehot = label_binarize(all_labels, classes=range(num_classes)) # (N, C)
        
        # Stack features and targets
        # data = np.hstack([all_features, y_onehot]) # (N, D+C)
        # corr = np.corrcoef(data, rowvar=False) # (D+C, D+C)
        
        # We only care about the correlation between Features and Classes.
        # Submatrix: corr[:D, D:] -> Shape (D, C)
        
        # Simpler manual calculation to save memory/compute:
        # Corr(X, Y) = Cov(X, Y) / (Std(X)*Std(Y))
        # This can be vectorized.
        
        print("Calculating feature correlations...")
        features_centered = all_features - all_features.mean(axis=0)
        y_centered = y_onehot - y_onehot.mean(axis=0)
        
        # Covariance: (X.T @ Y) / (N-1) -> Shape (D, C)
        cov = (features_centered.T @ y_centered) / (all_features.shape[0] - 1)
        
        std_feat = all_features.std(axis=0, ddof=1)
        std_y = y_onehot.std(axis=0, ddof=1)
        
        # Avoid div by zero
        std_feat[std_feat == 0] = 1e-8
        std_y[std_y == 0] = 1e-8
        
        # Outer product of stds -> (D, C)
        denom = std_feat[:, None] @ std_y[None, :]
        
        corr_matrix = cov / denom # Shape (D, C) -> D rows (features), C cols (classes)
        
        # User requested: X=Features, Y=Cell Type.
        # Current: Rows=Features, Cols=Classes.
        # So we transpose to Get Rows=Classes, Cols=Features.
        heatmap_data = corr_matrix.T # Shape (C, D)
        
        plt.figure(figsize=(15, 8))
        # no x-axis labels requested
        sns.heatmap(
            heatmap_data,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=False,
            yticklabels=class_names,
        )
        plt.xlabel('Features (Extracted)')
        plt.ylabel('Cell Type')
        plt.title(f'Feature-Histology Correlation ({model_name})')
        
        heatmap_path = os.path.join(output_dir, 'feature_corr.png')
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Saved Feature Correlation Heatmap to {heatmap_path}")

    return acc, f1_macro, cm
