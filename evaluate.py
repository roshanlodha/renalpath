import torch
import numpy as np
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

def evaluate_model(model, dataloader, device, num_classes=5, class_names=None, output_dir='.'):
    """
    Evaluates the model on the test set and computes metrics:
    ACC, Balanced ACC, MCC, F1 (Macro).
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # --- Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    
    # Detailed Report
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    print("Class-wise Metrics:")
    for i in range(num_classes):
        c_name = class_names[i] if class_names is not None else str(i)
        if i < len(precision):
            print(f"Class {c_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1_per_class[i]:.4f}")
    
    # AUC-ROC (One-vs-Rest)
    try:
        # Check if all classes present in test set for ROC
        present_classes = np.unique(all_labels)
        if len(present_classes) < num_classes:
             print("Warning: Not all classes present in test set, AUC might be approximate.")
             
        y_test_bin = label_binarize(all_labels, classes=range(num_classes))
        # Handle case where only 1 class is present (though unlikely with Stratified Split unless small data)
        if y_test_bin.shape[1] == num_classes:
            auc = roc_auc_score(y_test_bin, all_probs, multi_class='ovr', average='macro')
            print(f"AUC-ROC (Macro OvR): {auc:.4f}")
        else:
            print("Skipping AUC-ROC due to class mismatch in binarization.")
            
    except Exception as e:
        print(f"Could not calculate AUC-ROC: {e}")
        
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names is not None else "auto",
                yticklabels=class_names if class_names is not None else "auto")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    
    return acc, f1_macro, cm
