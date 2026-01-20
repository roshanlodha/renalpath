import os
import cv2
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def preprocess_images(raw_data_dir, output_dir, metadata_csv):
    """
    Filters and resizes images, then saves them to output_dir.
    Returns a DataFrame with valid image paths and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    processed_metadata_path = os.path.join(output_dir, 'processed_metadata.csv')
    train_split_path = os.path.join(output_dir, 'train_split.csv')
    val_split_path = os.path.join(output_dir, 'val_split.csv')
    test_split_path = os.path.join(output_dir, 'test_split.csv')

    # Caching check: if processed images already exist, reuse the manifest to enable re-splitting.
    if os.path.exists(processed_metadata_path):
        print(f"Processed images found in {output_dir}. Skipping preprocessing.")
        return pd.read_csv(processed_metadata_path)

    # Backwards-compatible fallback (older runs): reconstruct manifest from existing splits
    if os.path.exists(train_split_path) and os.path.exists(val_split_path) and os.path.exists(test_split_path):
        print(f"Processed splits found in {output_dir} but no manifest; reconstructing processed metadata.")
        df_train = pd.read_csv(train_split_path)
        df_val = pd.read_csv(val_split_path)
        df_test = pd.read_csv(test_split_path)
        df_processed = pd.concat([df_train, df_val, df_test], ignore_index=True)[['img_path', 'Patient ID', 'Class']]
        df_processed.to_csv(processed_metadata_path, index=False)
        return df_processed

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
        
    processed_df = pd.DataFrame(valid_data)
    processed_df.to_csv(processed_metadata_path, index=False)
    return processed_df

def create_splits(
    df,
    output_dir,
    *,
    trainval_fraction=0.5,
    val_fraction_of_trainval=0.2,
    random_state=42,
    force=False,
):
    """
    Performs patient-level stratified split:
    - Train+Val: 50%
    - Test: 50%
    - Val: 20% of Train+Val (i.e., 10% total)
    """
    os.makedirs(output_dir, exist_ok=True)

    train_split_path = os.path.join(output_dir, 'train_split.csv')
    val_split_path = os.path.join(output_dir, 'val_split.csv')
    test_split_path = os.path.join(output_dir, 'test_split.csv')
    classes_path = os.path.join(output_dir, 'classes.npy')
    split_config_path = os.path.join(output_dir, 'split_config.json')

    desired_config = {
        'trainval_fraction': float(trainval_fraction),
        'test_fraction': float(1.0 - trainval_fraction),
        'val_fraction_of_trainval': float(val_fraction_of_trainval),
        'random_state': int(random_state),
        'group_column': 'Patient ID',
        'stratify_column': 'Class',
    }

    if not force and os.path.exists(split_config_path) and \
       os.path.exists(train_split_path) and os.path.exists(val_split_path) and os.path.exists(test_split_path) and os.path.exists(classes_path):
        try:
            with open(split_config_path, 'r') as f:
                existing = json.load(f)
        except Exception:
            existing = {}

        keys = desired_config.keys()
        if all(existing.get(k) == desired_config[k] for k in keys):
            print("Splits already exist and match split_config.json. Skipping split creation.")
            return

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Class'])
    
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label Mapping: {label_map}")

    # Patient-level stratified split by patient label
    # Derive a single label per patient (mode of the patient's images).
    patient_labels = (
        df.groupby('Patient ID')['Class']
        .agg(lambda s: s.value_counts().idxmax())
    )
    if (df.groupby('Patient ID')['Class'].nunique() > 1).any():
        mixed = (df.groupby('Patient ID')['Class'].nunique() > 1).sum()
        print(f"WARNING: Detected {mixed} patients with multiple classes; using per-patient majority label for stratification.")

    class_to_label = {cls_name: int(i) for i, cls_name in enumerate(le.classes_)}
    patient_y = patient_labels.map(class_to_label)

    patient_ids = patient_labels.index.to_numpy()

    trainval_patients, test_patients = train_test_split(
        patient_ids,
        test_size=(1.0 - trainval_fraction),
        stratify=patient_y.to_numpy(),
        random_state=random_state,
        shuffle=True,
    )

    trainval_y = patient_y.loc[trainval_patients]
    train_patients, val_patients = train_test_split(
        trainval_patients,
        test_size=val_fraction_of_trainval,
        stratify=trainval_y.to_numpy(),
        random_state=random_state,
        shuffle=True,
    )

    df['split'] = 'train'
    df.loc[df['Patient ID'].isin(val_patients), 'split'] = 'val'
    df.loc[df['Patient ID'].isin(test_patients), 'split'] = 'test'
    
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

    with open(split_config_path, 'w') as f:
        json.dump(desired_config, f, indent=2, sort_keys=True)
