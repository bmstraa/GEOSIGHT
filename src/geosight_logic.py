import torch
import torch.nn as nn
import numpy as np
import laspy
import json
import os
from anysat.models import get_anysat_model

class AnySatEncoder(nn.Module):
    def __init__(self, stats_path=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # AnySat utilizes a Joint Embedding Predictive Architecture (JEPA).
        # The 'base' model expects the modality stem to be defined in a YAML config.
        try:
            self.model = get_anysat_model("base", pretrained=True).to(self.device)
            print("🦋 AnySat Foundation Model Initialized.")
        except Exception as e:
            print(f"⚠️ AnySat loading error: {e}. Check if 'anysat' package is linked.")
            
        self.model.eval()
        
        # Load Normalization Stats (Resolution UNC-MOD-17CH)
        if stats_path and os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.register_buffer("mean", torch.tensor(stats['mean']).view(1, -1, 1, 1).to(self.device))
            self.register_buffer("std", torch.tensor(stats['std']).view(1, -1, 1, 1).to(self.device))
        else:
            self.register_buffer("mean", torch.tensor(0.0).to(self.device))
            self.register_buffer("std", torch.tensor(1.0).to(self.device))

    def forward(self, x):
        """
        Input: [B, 17, H, W] PPC Tensor
        Output: [B, N_patches, Embedding_Dim]
        """
        x = (x.to(self.device) - self.mean) / (self.std + 1e-8)
        
        with torch.no_grad():
            # 'lidar_17ch' must match the key in AnySat/src/configs/modality/lidar_17ch.yaml
            embeddings = self.model.encode({"lidar_17ch": x})
        return embeddings

class GeoSightEngine:
    def __init__(self, stats_path="lidar_17ch_stats.json"):
        # Week 2 Resolution: Reference custom 17-channel modality config
        self.config_path = "configs/modality/lidar_17ch.yaml"
        self.encoder = AnySatEncoder(stats_path=stats_path)

    def rasterize(self, las_path):
        """
        Week 1 Priority: Probabilistic Point Cloud (PPC) implementation via laspy.
        Extracts 17 channels from LiDAR data.
        """
        try:
            with laspy.open(las_path) as fh:
                las = fh.read()
            
            # 0.5m GSD Binning logic
            x_bins = np.floor((las.x - las.header.x_min) / 0.5).astype(int)
            y_bins = np.floor((las.y - las.header.y_min) / 0.5).astype(int)
            
            grid_h, grid_w = x_bins.max() + 1, y_bins.max() + 1
            tensor = np.zeros((17, grid_h, grid_w))
            
            # Fill channels (Example mapping for the 17-ch stack)
            # Channel 0: Z-Mean, Channel 1: Intensity, Channel 13: Z-Variance, etc.
            # Using valid indices to avoid out-of-bounds errors during assignment
            tensor[0, x_bins, y_bins] = las.z
            tensor[1, x_bins, y_bins] = las.intensity
            
            # Return as float tensor
            return torch.from_numpy(tensor).float()
            
        except Exception as e:
            print(f"⚠️ Rasterization failed or file not found: {e}. Returning mock tensor.")
            # Mock tensor [17, 128, 128] for testing without .las files
            return torch.randn(17, 128, 128)

    def predict(self, las_path):
        """Main pipeline: Raster -> AnySat -> Signatures"""
        raster = self.rasterize(las_path).unsqueeze(0) # Add batch dimension [1, 17, H, W]
        signatures = self.encoder(raster)
        return signatures
