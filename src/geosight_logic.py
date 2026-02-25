import torch
import torch.nn as nn
import json
import os
from anysat.models import get_anysat_model

class AnySatEncoder(nn.Module):
    def __init__(self, stats_path=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load core AnySat with custom modality support
        # Note: 'base' refers to the ViT-B backbone
        self.model = get_anysat_model("base", pretrained=True).to(self.device)
        self.model.eval()
        
        # Normalization Layer
        if stats_path and os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.mean = torch.tensor(stats['mean']).view(1, -1, 1, 1).to(self.device)
            self.std = torch.tensor(stats['std']).view(1, -1, 1, 1).to(self.device)
        else:
            self.mean, self.std = 0.0, 1.0 # Fallback

    def forward(self, x):
        """
        x: [B, 17, H, W] - The 17-channel PPC stack
        """
        # 1. Normalize
        x = (x - self.mean) / (self.std + 1e-8)
        
        # 2. Extract Embeddings
        with torch.no_grad():
            # We map the input to the key 'lidar_17ch' defined in our config
            out = self.model.encode({"lidar_17ch": x})
        return out # Shape: [B, N_patches, Embedding_Dim]
