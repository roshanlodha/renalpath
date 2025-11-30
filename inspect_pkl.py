import torch
import pickle
import sys

try:
    data = torch.load('GSViT.pkl', map_location='cpu')
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        if 'state_dict' in data:
            print("Contains state_dict")
        if 'model' in data:
            print(f"Model type: {type(data['model'])}")
    else:
        print(f"Content: {data}")
except Exception as e:
    print(f"Error loading with torch: {e}")
    try:
        with open('GSViT.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded with pickle. Type: {type(data)}")
    except Exception as e2:
        print(f"Error loading with pickle: {e2}")
