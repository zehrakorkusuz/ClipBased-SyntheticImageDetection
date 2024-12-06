# experiments/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class CLIPPairDataset(Dataset):
    def __init__(self, csv_path, base_dir=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir or ''
        self.transform = transform or self._default_transform()
        
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def __len__(self):
        # Each row contains both real and fake image, so length is doubled
        return len(self.df) * 2
    
    def __getitem__(self, idx):
        # Determine if we're accessing real (0) or fake (1) image
        is_fake = idx >= len(self.df)
        row_idx = idx % len(self.df)
        
        row = self.df.iloc[row_idx]
        
        # Select filename based on real/fake
        filename = row['filename1'] if is_fake else row['filename0']
        filepath = os.path.join(self.base_dir, 'train_set', filename) 

        
        try:
            image = Image.open(filepath).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            # Return a blank image in case of error
            image = torch.zeros(3, 224, 224)
            
        return {
            'image': image,
            'label': int(is_fake),
            'caption': row['caption']
        }
    
class ValidationDataset(Dataset):
    def __init__(self, csv_path, base_dir=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir or ''
        self.transform = transform or self._default_transform()
        
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.base_dir, 'test_set', row['filename'])
        
        # Determine if image is real based on 'typ' field
        is_fake = not 'real' in row['typ'].lower()
        
        try:
            image = Image.open(filepath).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            image = torch.zeros(3, 224, 224)
            
        return {
            'image': image,
            'label': int(is_fake),  # 0 for real, 1 for fake
            'filename': row['filename']
        }