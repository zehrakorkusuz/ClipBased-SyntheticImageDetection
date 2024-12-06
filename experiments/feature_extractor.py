## to move to utils

import torch
from networks.openclipnet import OpenClipLinear
import numpy as np
import torch.nn.functional as F

class CLIPFeatureExtractor:
    def __init__(self, pretrain='clipL14commonpool', device='cuda'):
        # Initialize with next_to_last=True to get pre-projection features
        self.model = OpenClipLinear(pretrain=pretrain, next_to_last=True)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.clip_model = self.model.bb[0]  # Access underlying CLIP model

    def extract_features(self, image, text=None):
        with torch.no_grad():
            # Get pre-projection image features
            img_features = self.model.forward_features(image)
            
            if text is not None:
                # Get text features using CLIP's text encoder
                text_tokens = self.clip_model.encode_text(text)
                if self.model.normalize:
                    text_tokens = F.normalize(text_tokens, dim=-1)
                return {
                    'image': img_features,
                    'text': text_tokens,
                    'joint': torch.cat([img_features, text_tokens], dim=-1)
                }
            return img_features