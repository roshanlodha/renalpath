import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random

from preprocess import preprocess_images, create_splits
from dataset import TumorDataset, get_transforms
from models import ResNet50_Classifier, GSViT_Classifier, ViT_Classifier
from train import train_model, get_weighted_dataloader, save_training_curves
from evaluate import evaluate_model
from loss import FocalLoss

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DATA_DIR = 'data'


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Tumor Classification Pipeline')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'evaluate', 'dry_run'], required=True, help='Pipeline mode')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'vit', 'gsvit'], default='resnet', help='Model architecture')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to data directory')
    parser.add_argument('--processed_dir', type=str, default=os.path.join(DATA_DIR, 'processed'), help='Path to processed data')
    parser.add_argument('--metadata_csv', type=str, default=os.path.join(DATA_DIR, 'metadata.csv'), help='Path to metadata CSV')
    parser.add_argument('--gsvit_path', type=str, default='models/GSViT.pkl', help='Path to GSViT pickle file')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == 'dry_run':
        print("Running dry run verification...")
        # Verify ResNet
        print("Checking ResNet-50...")
        model = ResNet50_Classifier(num_classes=5)
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        print(f"ResNet output shape: {out.shape}")
        with torch.no_grad():
            logits, feats = model(dummy_input, return_features=True)
        print(f"ResNet features shape: {None if feats is None else tuple(feats.shape)}")

        # Verify ViT
        print("Checking ViT-B/16...")
        model = ViT_Classifier(num_classes=5)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        print(f"ViT output shape: {out.shape}")
        with torch.no_grad():
            logits, feats = model(dummy_input, return_features=True)
        print(f"ViT features shape: {None if feats is None else tuple(feats.shape)}")
        
        # Verify GSViT
        print(f"Checking GSViT from {args.gsvit_path}...")
        if os.path.exists(args.gsvit_path):
            try:
                model = GSViT_Classifier(args.gsvit_path, num_classes=5)
                # GSViT might expect BGR, but shape is same
                model.eval()
                with torch.no_grad():
                    out = model(dummy_input)
                print(f"GSViT output shape: {out.shape}")
                with torch.no_grad():
                    logits, feats = model(dummy_input, return_features=True)
                print(f"GSViT features shape: {None if feats is None else tuple(feats.shape)}")
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
        model = ResNet50_Classifier(num_classes=num_classes)
        model_name_str = 'resnet'
    elif args.model_type == 'vit':
        model = ViT_Classifier(num_classes=num_classes)
        model_name_str = 'vit'
    else:
        model = GSViT_Classifier(args.gsvit_path, num_classes=num_classes)
        model_name_str = 'gsvit'
        
    if args.mode == 'train':
        # Caching: Check if training is needed
        best_model_path = os.path.join(args.output_dir, f'best_{args.model_type}.pth')
        if os.path.exists(best_model_path):
             print(f"Trained model found at {best_model_path}. Skipping training.")
             return

        # Datasets
        train_csv = os.path.join(args.processed_dir, 'train_split.csv')
        val_csv = os.path.join(args.processed_dir, 'val_split.csv')
        
        train_dataset = TumorDataset(train_csv, root_dir=args.processed_dir, transform=get_transforms('train', model_name_str), model_name=model_name_str)
        val_dataset = TumorDataset(val_csv, root_dir=args.processed_dir, transform=get_transforms('val', model_name_str), model_name=model_name_str)
        
        # Use WeightedRandomSampler for training
        train_loader = get_weighted_dataloader(train_dataset, batch_size=args.batch_size, num_classes=num_classes, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # Loss: FocalLoss with alpha=1
        # The prompt asked for "FocalLoss class manually... -(alpha * (1 - pt)^gamma * log(pt))"
        # We can pass alpha if we want class weighting there too, or just 1.0. 
        # Using 1.0 based on "Formula: -(alpha...)" without specifying dynamic alpha calc, 
        # plus we are using WeightedRandomSampler which balances sampling already.
        # Balancing BOTH sampling AND loss weights is usually redundant/harmful.
        # Since Sampler is requested explicitly for balancing, we stick to FocalLoss for hard examples (gamma).
        criterion = FocalLoss(alpha=1, gamma=2.0)
        
        model, history = train_model(model, dataloaders, device, num_epochs=args.epochs, criterion=criterion)
        
        # Save model
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_{args.model_type}.pth'))

        analysis_dir = os.path.join('analysis', model_name_str)
        os.makedirs(analysis_dir, exist_ok=True)
        save_training_curves(history, os.path.join(analysis_dir, 'training_curves.png'), model_name=model_name_str)
        
    elif args.mode == 'evaluate':
        test_csv = os.path.join(args.processed_dir, 'test_split.csv')
        test_dataset = TumorDataset(test_csv, root_dir=args.processed_dir, transform=get_transforms('val', model_name_str), model_name=model_name_str)
        dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Load best model
        model_path = os.path.join(args.output_dir, f'best_{args.model_type}.pth')
        if not os.path.exists(model_path):
             print(f"Model file not found at {model_path}. Please train first.")
             return

        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Output analysis to a separate folder
        analysis_dir = os.path.join('analysis', model_name_str)
        train_prevalence = None
        train_csv = os.path.join(args.processed_dir, 'train_split.csv')
        if os.path.exists(train_csv):
            train_df = pd.read_csv(train_csv)
            if 'label_encoded' in train_df.columns:
                counts = np.bincount(train_df['label_encoded'].astype(int), minlength=num_classes)
                total = counts.sum()
                if total > 0:
                    train_prevalence = counts / total

        evaluate_model(
            model,
            dataloader,
            device,
            num_classes=num_classes,
            class_names=class_names,
            output_dir=analysis_dir,
            model_name=model_name_str,
            train_class_prevalence=train_prevalence,
        )

if __name__ == "__main__":
    main()
