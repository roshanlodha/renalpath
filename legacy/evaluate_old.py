import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

def evaluate_model(model, dataloader, device, num_classes=5, class_names=None, output_dir='.'):
    """
    Evaluates the model on the test set and computes metrics.
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
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    print(f"Overall Accuracy: {acc:.4f}")
    print("Class-wise Metrics:")
    for i in range(num_classes):
        c_name = class_names[i] if class_names is not None else str(i)
        print(f"Class {c_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
        
    # AUC-ROC (One-vs-Rest)
    try:
        y_test_bin = label_binarize(all_labels, classes=range(num_classes))
        auc = roc_auc_score(y_test_bin, all_probs, multi_class='ovr', average='macro')
        print(f"AUC-ROC (Macro OvR): {auc:.4f}")
    except ValueError:
        print("Could not calculate AUC-ROC (likely missing classes in test set)")
        
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()
    
    return acc, f1, cm
