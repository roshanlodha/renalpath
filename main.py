import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random

from preprocess import preprocess_images, create_splits
from dataset import TumorDataset, get_transforms
from resnet_model import get_resnet_model
from gsvit_model import get_gsvit_model
from train import train_model
from evaluate import evaluate_model

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Tumor Classification Pipeline')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'evaluate', 'dry_run'], required=True, help='Pipeline mode')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'gsvit'], default='resnet', help='Model architecture')
    parser.add_argument('--data_dir', type=str, default='Images', help='Path to raw data')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--metadata_csv', type=str, default='data/metadata.csv', help='Path to metadata CSV')
    parser.add_argument('--gsvit_path', type=str, default='GSViT.pkl', help='Path to GSViT pickle file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == 'dry_run':
        print("Running dry run verification...")
        # Verify ResNet
        print("Checking ResNet-50...")
        model = get_resnet_model(num_classes=5)
        dummy_input = torch.randn(1, 3, 224, 224)
        out = model(dummy_input)
        print(f"ResNet output shape: {out.shape}")
        
        # Verify GSViT
        print(f"Checking GSViT from {args.gsvit_path}...")
        if os.path.exists(args.gsvit_path):
            try:
                model = get_gsvit_model(args.gsvit_path, num_classes=5)
                # GSViT might expect BGR, but shape is same
                out = model(dummy_input)
                print(f"GSViT output shape: {out.shape}")
            except Exception as e:
                print(f"GSViT check failed: {e}")
        else:
            print(f"GSViT file not found at {args.gsvit_path}, skipping load check.")
            
        print("Dry run completed.")
        return

    if args.mode == 'preprocess':
        df = preprocess_images(args.data_dir, args.processed_dir, args.metadata_csv)
        create_splits(df, args.processed_dir)
        return

    # Load classes
    classes_path = os.path.join(args.processed_dir, 'classes.npy')
    if os.path.exists(classes_path):
        class_names = np.load(classes_path, allow_pickle=True)
        num_classes = len(class_names)
    else:
        print("Classes file not found, assuming 5 classes.")
        num_classes = 5
        class_names = [str(i) for i in range(5)]

    # Model selection
    if args.model_type == 'resnet':
        model = get_resnet_model(num_classes=num_classes)
        is_gsvit = False
    else:
        model = get_gsvit_model(args.gsvit_path, num_classes=num_classes)
        is_gsvit = True
        
    if args.mode == 'train':
        # Datasets
        train_csv = os.path.join(args.processed_dir, 'train_split.csv')
        val_csv = os.path.join(args.processed_dir, 'val_split.csv')
        
        train_dataset = TumorDataset(train_csv, transform=get_transforms('train', is_gsvit), is_gsvit=is_gsvit)
        val_dataset = TumorDataset(val_csv, transform=get_transforms('val', is_gsvit), is_gsvit=is_gsvit)
        
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        }
        
        # Calculate class weights
        # "weights were set based on the inverse class frequency in the training data"
        labels = train_dataset.data['label_encoded'].values
        class_counts = np.bincount(labels, minlength=num_classes)
        total_samples = len(labels)
        class_weights = torch.tensor(total_samples / (num_classes * class_counts), dtype=torch.float)
        print(f"Class weights: {class_weights}")
        
        model, history = train_model(model, dataloaders, device, num_epochs=args.epochs, class_weights=class_weights)
        
        # Save model
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_{args.model_type}.pth'))
        
    elif args.mode == 'evaluate':
        test_csv = os.path.join(args.processed_dir, 'test_split.csv')
        test_dataset = TumorDataset(test_csv, transform=get_transforms('val', is_gsvit), is_gsvit=is_gsvit)
        dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Load best model
        model_path = os.path.join(args.output_dir, f'best_{args.model_type}.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        evaluate_model(model, dataloader, device, num_classes=num_classes, class_names=class_names, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
