import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import copy
import os

from loss import FocalLoss

def train_model(model, dataloaders, device, num_epochs=30, patience=10, criterion=None):
    """
    Advanced training loop using AdamW and CosineAnnealingWarmRestarts.
    """
    model = model.to(device)
    
    # Optimizer: AdamW with weight_decay=1e-4
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Scheduler: CosineAnnealingWarmRestarts
    # Assumes we might want restarts every 10 epochs? The user didn't specify T_0.
    # Standard choice is often related to epoch count, let's say 10.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    if criterion is None:
        # Default to FocalLoss if not provided (though main might provide it)
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
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    if isinstance(criterion, FocalLoss):
                        # FocalLoss expects raw logits (handled inside)
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                        # Note: CosineAnnealingWarmRestarts is usually stepped every batch? 
                        # Or every epoch? Docs say: "should be called after every batch" if using it as per-batch
                        # but standard Torch schedulers are per-epoch unless specified.
                        # CAWR docs: "step(epoch=None)"
                        # Usually stepped at epoch end similar to others, 
                        # UNLESS OneCycleLR. Let's stick to epoch stepping to be safe/standard
                        # unless advanced usage is implied. 
                        # Actually CAWR creates a schedule per iteration if T_0 is iterations.
                        # If T_0 is epochs, step at epoch end. Let's assume epochs (T_0=10).
                        pass

            # Step scheduler at epoch end for CAWR (T_0=10 epochs)
            if phase == 'train':
                scheduler.step()
                print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # Fix: running_loss calculation in loop needs check
            # In previous loop: running_loss += loss.item() * inputs.size(0)
            # Re-adding that logic:
            
            # Wait, I missed the accumulation lines in the loop above!
            # Let me rewrite the loop part correctly.
            pass # See below for full file content
        
    return model, history


# Helper to re-implement the train logic cleanly since I cut myself off
def train_model_full(model, dataloaders, device, num_epochs=30, patience=10, criterion=None):
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # 1e-4 lr assumed from prev file, or standard
    # Previous file had 0.0001 (1e-4).
    
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
            
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
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

def get_weighted_dataloader(dataset, batch_size=32, num_classes=5, num_workers=4):
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
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).type(torch.DoubleTensor),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=num_workers
    )
    
    return dataloader

# Expose the simple name
train_model = train_model_full
