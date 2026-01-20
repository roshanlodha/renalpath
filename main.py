import argparse
import os
import torch
import configparser
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


def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return None
    config.read(config_path)
    return config


def main():
    set_seed(42)
    
    # Load Config
    config = load_config()
    
    # Defaults from config if available
    defaults = {}
    if config:
        defaults['batch_size'] = config.getint('Hyperparameters', 'batch_size', fallback=32)
        defaults['epochs'] = config.getint('Hyperparameters', 'epochs', fallback=10)
        defaults['learning_rate'] = config.getfloat('Hyperparameters', 'learning_rate', fallback=1e-4)
        defaults['weight_decay'] = config.getfloat('Hyperparameters', 'weight_decay', fallback=1e-4)
        defaults['patience'] = config.getint('Hyperparameters', 'patience', fallback=10)
        defaults['num_workers'] = config.getint('Hyperparameters', 'num_workers', fallback=4)
        
        defaults['model_type'] = config.get('Model', 'model_type', fallback='resnet')
        defaults['gsvit_path'] = config.get('Model', 'gsvit_path', fallback='models/GSViT.pkl')
        
        defaults['data_dir'] = config.get('Data', 'data_dir', fallback='data')
        defaults['processed_dir'] = config.get('Data', 'processed_dir', fallback=os.path.join(defaults['data_dir'], 'processed'))
        defaults['metadata_csv'] = config.get('Data', 'metadata_csv', fallback=os.path.join(defaults['data_dir'], 'metadata.csv'))
        defaults['upsample'] = config.getboolean('Data', 'upsample', fallback=True)
    else:
        # Hardcoded defaults if no config
        defaults = {
            'batch_size': 32, 'epochs': 10, 'learning_rate': 1e-4, 'weight_decay': 1e-4, 'patience': 10, 'num_workers': 4,
            'model_type': 'resnet', 'gsvit_path': 'models/GSViT.pkl',
            'data_dir': 'data', 'processed_dir': 'data/processed', 'metadata_csv': 'data/metadata.csv', 'upsample': True
        }

    parser = argparse.ArgumentParser(description='Tumor Classification Pipeline')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'evaluate', 'dry_run'], required=True, help='Pipeline mode')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'vit', 'gsvit'], default=defaults['model_type'], help='Model architecture')
    parser.add_argument('--data_dir', type=str, default=defaults['data_dir'], help='Path to data directory')
    parser.add_argument('--processed_dir', type=str, default=defaults['processed_dir'], help='Path to processed data')
    parser.add_argument('--metadata_csv', type=str, default=defaults['metadata_csv'], help='Path to metadata CSV')
    parser.add_argument('--gsvit_path', type=str, default=defaults['gsvit_path'], help='Path to GSViT pickle file')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'], help='Batch size')
    parser.add_argument('--epochs', type=int, default=defaults['epochs'], help='Number of epochs')
    
    # Extra args not in previous CLI but useful to override
    parser.add_argument('--lr', type=float, default=defaults['learning_rate'], help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=defaults['weight_decay'], help='Weight decay')
    parser.add_argument('--patience', type=int, default=defaults['patience'], help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Ensure processed_dir exists for split processing
    if not os.path.exists(args.processed_dir) and args.mode == 'preprocess':
         os.makedirs(args.processed_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check classes if not preprocessing
    # Determine num_classes
    num_classes = None
    if args.mode != 'preprocess':
        classes_path = os.path.join(args.processed_dir, 'classes.npy')
        if os.path.exists(classes_path):
            class_names = np.load(classes_path, allow_pickle=True)
            num_classes = len(class_names)
            print(f"Detected {num_classes} classes: {class_names}")
        else:
            raise FileNotFoundError(f"Classes file not found at {classes_path}. Please run preprocessing first.")

    if args.mode == 'dry_run':
        print("Running dry run verification...")
        # Verify ResNet
        print("Checking ResNet-50...")
        model = ResNet50_Classifier(num_classes=num_classes)
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        print(f"ResNet output shape: {out.shape}")
        
        # Verify ViT
        print("Checking ViT-B/16...")
        model = ViT_Classifier(num_classes=num_classes)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        print(f"ViT output shape: {out.shape}")
        
        if args.model_type == 'gsvit': 
             if os.path.exists(args.gsvit_path):
                print(f"Checking GSViT from {args.gsvit_path}...")
                try:
                    model = GSViT_Classifier(args.gsvit_path, num_classes=num_classes)
                    model.eval()
                    with torch.no_grad():
                        out = model(dummy_input)
                    print(f"GSViT output shape: {out.shape}")
                except Exception as e:
                    print(f"GSViT check failed: {e}")
        
        print("Dry run completed.")
        return

    if args.mode == 'preprocess':
        df = preprocess_images(args.data_dir, args.processed_dir, args.metadata_csv)
        create_splits(df, args.processed_dir)
        return

    # Model initialization
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
             print(f"Trained model found at {best_model_path}. Skipping training. Delete file to retrain.")
             pass   

        # Datasets
        train_csv = os.path.join(args.processed_dir, 'train_split.csv')
        val_csv = os.path.join(args.processed_dir, 'val_split.csv')
        
        train_dataset = TumorDataset(train_csv, root_dir=args.processed_dir, transform=get_transforms('train', model_name_str), model_name=model_name_str)
        val_dataset = TumorDataset(val_csv, root_dir=args.processed_dir, transform=get_transforms('val', model_name_str), model_name=model_name_str)
        
        # Use WeightedRandomSampler for training if upsample is True
        print(f"Upsampling enabled: {defaults['upsample']}")
        train_loader = get_weighted_dataloader(
            train_dataset, 
            batch_size=args.batch_size, 
            num_classes=num_classes, 
            num_workers=defaults['num_workers'],
            upsample_to_max=defaults['upsample']
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=defaults['num_workers'])
        
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # Loss: FocalLoss
        criterion = FocalLoss(alpha=1, gamma=2.0)
        
        model, history = train_model(
            model, 
            dataloaders, 
            device, 
            num_epochs=args.epochs, 
            criterion=criterion,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience
        )
        
        # Save model
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_{args.model_type}.pth'))

        analysis_dir = os.path.join('analysis', model_name_str)
        os.makedirs(analysis_dir, exist_ok=True)
        save_training_curves(history, os.path.join(analysis_dir, 'training_curves.png'), model_name=model_name_str)
        
    elif args.mode == 'evaluate':
        test_csv = os.path.join(args.processed_dir, 'test_split.csv')
        test_dataset = TumorDataset(test_csv, root_dir=args.processed_dir, transform=get_transforms('val', model_name_str), model_name=model_name_str)
        dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=defaults['num_workers'])
        
        # Load best model
        model_path = os.path.join(args.output_dir, f'best_{args.model_type}.pth')
        if not os.path.exists(model_path):
             print(f"Model file not found at {model_path}. Please train first.")
             return

        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Output analysis
        analysis_dir = os.path.join('analysis', model_name_str)
        # Load train prevalence for balanced accuracy reference
        train_prevalence = None
        train_csv = os.path.join(args.processed_dir, 'train_split.csv')
        if os.path.exists(train_csv):
            try:
                train_df = pd.read_csv(train_csv)
                if 'label_encoded' in train_df.columns:
                    counts = np.bincount(train_df['label_encoded'].astype(int), minlength=num_classes)
                    total = counts.sum()
                    if total > 0:
                        train_prevalence = counts / total
            except:
                pass

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
