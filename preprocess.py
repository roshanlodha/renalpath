import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def preprocess_images(raw_data_dir, output_dir, metadata_csv):
    """
    Filters and resizes images, then saves them to output_dir.
    Returns a DataFrame with valid image paths and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(metadata_csv)
    valid_data = []
    
    print("Preprocessing images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Construct path - Assuming structure is raw_dir/Images/image_name based on metadata
        # Previous analysis showed metadata has 'img_path' column which likely includes 'Images/filename'
        # Let's check metadata columns again. 
        # Metadata columns: Patient ID,Image ID,Date,Class,img_path ("Images/0002...")
        
        # If img_path is relative to root, and raw_data_dir is root...
        img_full_path = os.path.join(raw_data_dir, row['img_path'])
        
        if not os.path.exists(img_full_path):
            # Try alternative if raw_data_dir points to 'Images' parent
            # If raw_data_dir is just the root, and img_path is 'Images/...', it works.
            continue
            
        img = cv2.imread(img_full_path)
        if img is None:
            continue
            
        # Thresholding: "mean pixel intensity below 20 and std below 10 was considered black"
        if img.mean() < 20 and img.std() < 10:
            continue
            
        # Resize to 224x224
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB (OpenCV is BGR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Save standardized image
        # Saving in output_dir/patient_id ensures we organize well, but metadata has full control.
        # Let's flatten or mimic structure. The legacy code flattened into patient folders.
        # Metadata row has 'Patient ID'.
        patient_id = str(row['Patient ID'])
        save_subdir = os.path.join(output_dir, patient_id)
        os.makedirs(save_subdir, exist_ok=True)
        
        # Extract filename
        fname = os.path.basename(row['img_path'])
        save_path = os.path.join(save_subdir, fname)
        
        # Save back as BGR for consistency if reading with cv2 later, or RGB if PIL.
        # dataset.py uses cv2.imread (BGR) then converts to RGB.
        # So saving as BGR is safest for cv2.imread compatibility.
        cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
        valid_data.append({
            'img_path': os.path.join(patient_id, fname), # Relative to output_dir
            'Patient ID': patient_id,
            'Class': row['Class']
        })
        
    return pd.DataFrame(valid_data)

def create_splits(df, output_dir):
    """
    Performs patient-level stratified split (80/10/10) using StratifiedGroupKFold.
    """
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Class'])
    
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label Mapping: {label_map}")
    
    # StratifiedGroupKFold
    # We need 80% Train, 10% Val, 10% Test.
    # Step 1: 5 splits -> 20% each. Take 1 fold as Temp (20%), rest as Train (80%).
    sgkf = StratifiedGroupKFold(n_splits=5)
    
    df['split'] = 'train'
    
    # Groups must be patient IDs
    groups = df['Patient ID']
    y = df['label_encoded']
    
    for train_idx, temp_idx in sgkf.split(df, y, groups=groups):
        df.loc[temp_idx, 'split'] = 'temp'
        break
        
    # Step 2: Split Temp (20%) into Val (10%) and Test (10%) -> 50/50 split
    df_temp = df[df['split'] == 'temp'].copy()
    
    if len(df_temp) > 0:
        sgkf_val = StratifiedGroupKFold(n_splits=2)
        
        # Reset index for split to return correct relative indices, or use logic
        # Easier to use index map
        for val_idx, test_idx in sgkf_val.split(df_temp, df_temp['label_encoded'], groups=df_temp['Patient ID']):
            # val_idx are integer indices into df_temp
            val_indices = df_temp.iloc[val_idx].index
            test_indices = df_temp.iloc[test_idx].index
            
            df.loc[val_indices, 'split'] = 'val'
            df.loc[test_indices, 'split'] = 'test'
            break
    
    # Save splits
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Verify group independence
    train_patients = set(train_df['Patient ID'])
    val_patients = set(val_df['Patient ID'])
    test_patients = set(test_df['Patient ID'])
    
    intersect_tv = train_patients.intersection(val_patients)
    intersect_tt = train_patients.intersection(test_patients)
    intersect_vt = val_patients.intersection(test_patients)
    
    if intersect_tv or intersect_tt or intersect_vt:
        print("WARNING: Patient leakage detected!")
    else:
        print("Patient independence verified.")
    
    train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
    
    np.save(os.path.join(output_dir, 'classes.npy'), le.classes_)
