import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
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
    3. AUROC Bar Plot (auroc.png)
    4. ROC Curve (roc_curve.png) [binary only]
    4. Feature Correlation Heatmap (feature_corr.png) [GSViT/ViT]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Move model to target device *before* calling eval(). Some third-party
    # modules override train()/eval() and may cache tensors that won't be moved
    # by .to(device) if created on CPU.
    model = model.to(device)
    model.eval()
    
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
            try:
                logits, features = model(inputs, return_features=True)
            except TypeError:
                # Compatibility fallback for models without return_features arg
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
    
    # --- Plot 2: AUPRC/AUROC Bar Plots ---
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    # NOTE: sklearn's label_binarize returns shape (N, 1) for binary problems.
    # We use a true one-hot encoding to keep a consistent (N, C) shape.
    if num_classes == 2:
        y_onehot = np.eye(2, dtype=int)[all_labels.astype(int)]
    else:
        y_onehot = label_binarize(all_labels, classes=list(range(num_classes)))

    auprc_scores = []
    auroc_scores = []
    for i in range(num_classes):
        ap = np.nan
        auc_val = np.nan
        try:
            ap = average_precision_score(y_onehot[:, i], all_probs[:, i])
        except ValueError as e:
            print(f"Skipping AUPRC for class {class_names[i]}: {e}")
        try:
            auc_val = roc_auc_score(y_onehot[:, i], all_probs[:, i])
        except ValueError as e:
            print(f"Skipping AUROC for class {class_names[i]}: {e}")

        auprc_scores.append(ap)
        auroc_scores.append(auc_val)

        if not np.isnan(ap):
            print(f"Class {class_names[i]} AUPRC: {ap:.4f}")
        if not np.isnan(auc_val):
            print(f"Class {class_names[i]} AUROC: {auc_val:.4f}")

    x = np.arange(num_classes)
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_classes))

    # === AUPRC Plot ===
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, np.nan_to_num(auprc_scores, nan=0.0), color=colors, label='AUPRC')

    for bar, score in zip(bars, auprc_scores):
        label = 'NA' if np.isnan(score) else f'{score:.2f}'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            label,
            ha='center',
            va='bottom',
            fontsize=10
        )

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
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x, class_names, rotation=45, ha='right')
    ax.legend(loc='upper right')

    auprc_path = os.path.join(output_dir, 'auprc.png')
    fig.tight_layout()
    fig.savefig(auprc_path, dpi=200)
    plt.close(fig)
    print(f"Saved AUPRC Plot to {auprc_path}")

    # === AUROC Plot ===
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, np.nan_to_num(auroc_scores, nan=0.0), color=colors, label='AUROC')

    for bar, score in zip(bars, auroc_scores):
        label = 'NA' if np.isnan(score) else f'{score:.2f}'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            label,
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Random Guess (0.5)')

    ax.set_title(f'AUROC per Class ({model_name})')
    ax.set_ylabel('AUROC Score')
    ax.set_xlabel('Cell Type')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x, class_names, rotation=45, ha='right')
    ax.legend(loc='upper right')

    auroc_path = os.path.join(output_dir, 'auroc.png')
    fig.tight_layout()
    fig.savefig(auroc_path, dpi=200)
    plt.close(fig)
    print(f"Saved AUROC Plot to {auroc_path}")

    auprc_macro = float(np.nanmean(auprc_scores)) if len(auprc_scores) else None
    auroc_macro = float(np.nanmean(auroc_scores)) if len(auroc_scores) else None

    # --- Plot 2b: ROC curve (binary only) ---
    binary_roc_auc = None
    roc_path = None
    if num_classes == 2 and all_probs.shape[1] >= 2:
        try:
            y_true = all_labels.astype(int)
            y_score = all_probs[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = roc_auc_score(y_true, y_score)
            binary_roc_auc = float(auc_val)

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(fpr, tpr, color='tab:blue', linewidth=2, label=f'AUROC = {auc_val:.3f}')
            ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1, label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve ({model_name})')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right')
            fig.tight_layout()

            roc_path = os.path.join(output_dir, 'roc_curve.png')
            fig.savefig(roc_path, dpi=200)
            plt.close(fig)
            print(f"Saved ROC Curve to {roc_path}")
        except ValueError as e:
            print(f"Skipping ROC curve plot: {e}")

    # --- Summary JSON ---
    per_class = {
        str(class_names[i]): {
            'auprc': None if np.isnan(auprc_scores[i]) else float(auprc_scores[i]),
            'auroc': None if np.isnan(auroc_scores[i]) else float(auroc_scores[i]),
        }
        for i in range(num_classes)
    }

    summary = {
        'model_name': str(model_name),
        'num_classes': int(num_classes),
        'class_names': [str(x) for x in class_names],
        'metrics': {
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'mcc': float(mcc),
            'f1_macro': float(f1_macro),
            'auprc_macro': auprc_macro,
            'auroc_macro': auroc_macro,
            'binary_roc_auc': binary_roc_auc,
        },
        'per_class': per_class,
        'artifacts': {
            'confusion_matrix_png': os.path.basename(cm_path),
            'auprc_png': os.path.basename(auprc_path),
            'auroc_png': os.path.basename(auroc_path),
            'roc_curve_png': (os.path.basename(roc_path) if roc_path else None),
            'feature_corr_png': ('feature_corr.png' if (model_name in {'gsvit', 'vit'} and len(all_features) > 0) else None),
        },
        'confusion_matrix': cm.tolist(),
    }

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Saved summary to {summary_path}")

    # --- Plot 3: Feature Correlation Heatmap (GSViT/ViT) ---
    if model_name in {'gsvit', 'vit'} and len(all_features) > 0:
        # X: Features (1000s), Y: Cell Types
        # Calculate Correlation: Cov(X, Y) / (Std(X)*Std(Y))
        # X: Features (D), Y: One-hot Classes (C)
        print("Calculating feature correlations...")
        y_onehot = label_binarize(all_labels, classes=range(num_classes)) # (N, C)
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
