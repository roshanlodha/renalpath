import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import shutil
from pathlib import Path
from tqdm import tqdm

def preprocess_images(raw_data_dir, output_dir, metadata_csv):
    """
    Filters and resizes images, then saves them to output_dir.
    Returns a DataFrame with valid image paths and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    # Assuming metadata_csv has 'patient_id', 'image_name', 'tumor_class'
    df = pd.read_csv(metadata_csv)
    
    valid_data = []
    
    print("Preprocessing images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(raw_data_dir, row['patient_id'], row['image_name'])
        
        if not os.path.exists(img_path):
            continue
            
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Thresholding heuristic
        # "mean pixel intensity below 20 and standard deviation below 10 was considered black"
        if img.mean() < 20 and img.std() < 10:
            continue
            
        # Resize to 224x224
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB (OpenCV is BGR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Save standardized image
        # Maintaining patient structure or flat? "saved for downstream modeling"
        # Let's save in a flat structure or patient structure in output_dir
        save_subdir = os.path.join(output_dir, row['patient_id'])
        os.makedirs(save_subdir, exist_ok=True)
        save_path = os.path.join(save_subdir, row['image_name'])
        
        # Save as RGB? cv2.imwrite expects BGR. 
        # "converted to RGB format. These standardized images were then saved"
        # If we save with cv2.imwrite, we should convert back to BGR or use PIL.
        # Let's use cv2 and convert back to BGR for saving to ensure consistency.
        cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
        valid_data.append({
            'path': save_path,
            'patient_id': row['patient_id'],
            'label': row['tumor_class']
        })
        
    return pd.DataFrame(valid_data)

def create_splits(df, output_dir):
    """
    Performs patient-level stratified split (80/10/10) and saves split CSVs.
    """
    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    # Save label mapping
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label Mapping: {label_map}")
    
    # Split: Train (80%), Val (10%), Test (10%)
    # We use StratifiedGroupKFold to ensure patient separation and stratified labels
    # First split: Train (80%) vs Temp (20%)
    sgkf = StratifiedGroupKFold(n_splits=5) # 1/5 = 20% for test+val
    
    df['split'] = 'train'
    
    # Get one fold for temp (val + test)
    for train_idx, temp_idx in sgkf.split(df, df['label_encoded'], groups=df['patient_id']):
        df.loc[temp_idx, 'split'] = 'temp'
        break
        
    # Split Temp (20%) into Val (10%) and Test (10%) -> 50/50 split of temp
    df_temp = df[df['split'] == 'temp']
    sgkf_val = StratifiedGroupKFold(n_splits=2)
    
    for val_idx, test_idx in sgkf_val.split(df_temp, df_temp['label_encoded'], groups=df_temp['patient_id']):
        # Map back to original indices
        val_original_idx = df_temp.iloc[val_idx].index
        test_original_idx = df_temp.iloc[test_idx].index
        
        df.loc[val_original_idx, 'split'] = 'val'
        df.loc[test_original_idx, 'split'] = 'test'
        break
        
    # Save splits
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
    
    # Save label encoder classes
    np.save(os.path.join(output_dir, 'classes.npy'), le.classes_)

if __name__ == "__main__":
    # Example usage (commented out as paths are not real)
    # raw_dir = "path/to/raw_images"
    # out_dir = "path/to/processed_images"
    # meta_csv = "path/to/metadata.csv"
    # df = preprocess_images(raw_dir, out_dir, meta_csv)
    # create_splits(df, out_dir)
    pass
