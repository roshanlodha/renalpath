import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import copy
import os

from loss import FocalLoss





def train_model(model, dataloaders, device, num_epochs=30, patience=10, criterion=None, learning_rate=1e-4, weight_decay=1e-4):
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    
    if criterion is None:
        criterion = FocalLoss(gamma=2.0)
        
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0
            
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                running_samples += batch_size
            
            if phase == 'train':
                scheduler.step()
                
            if running_samples == 0:
                epoch_loss = 0.0
                epoch_acc = torch.tensor(0.0, device=device)
            else:
                epoch_loss = running_loss / running_samples
                epoch_acc = running_corrects.double() / running_samples
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
            
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, history

def get_weighted_dataloader(
    dataset,
    batch_size=32,
    num_classes=5,
    num_workers=4,
    *,
    upsample_to_max=True,
):
    """
    Returns a DataLoader with WeightedRandomSampler.
    """
    targets = dataset.data['label_encoded'].values
    class_counts = np.bincount(targets, minlength=num_classes)
    
    # Weights: 1 / class_count
    # Handle bias against classes with 0 samples (though unlikely in train)
    class_weights = 1. / np.maximum(class_counts, 1)
    
    # Assign weight to each sample
    sample_weights = class_weights[targets]
    
    # Create Sampler
    present_counts = class_counts[class_counts > 0]
    if upsample_to_max and len(present_counts) > 0:
        max_count = int(present_counts.max())
        num_present_classes = int((class_counts > 0).sum())
        num_samples = max_count * num_present_classes
    else:
        num_samples = len(sample_weights)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).type(torch.DoubleTensor),
        num_samples=num_samples,
        replacement=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=num_workers
    )
    
    return dataloader

# --- Plotting ---
def save_training_curves(history, output_path, model_name='model'):
    if not history:
        return

    keys = ('train_loss', 'val_loss', 'train_acc', 'val_acc')
    if not all(k in history for k in keys):
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss ({model_name})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train')
    plt.plot(epochs, history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy ({model_name})')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

