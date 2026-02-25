import torch
import torch.nn as nn
import numpy as np
import laspy
import json
import os
from anysat.models import get_anysat_model

class AnySatEncoder(nn.Module):
    def __init__(self, stats_path="lidar_17ch_stats.json"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # AnySat utilizes a Joint Embedding Predictive Architecture (JEPA).
        # We load the base model; the custom stem is defined in the .yaml config.
        try:
            self.model = get_anysat_model("base", pretrained=True).to(self.device)
            print("🦋 AnySat Foundation Model Initialized.")
        except Exception as e:
            print(f"⚠️ AnySat loading error: {e}. Ensure local 'configs' are set.")
        
        self.model.eval()
        
        # Load Normalization Stats (Resolution UNC-MOD-17CH)
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.mean = torch.tensor(stats['mean']).view(1, -1, 1, 1).to(self.device)
            self.std = torch.tensor(stats['std']).view(1, -1, 1, 1).to(self.device)
        else:
            self.mean, self.std = 0.0, 1.0

    def forward(self, x):
        """
        Input: [B, 17, H, W] PPC Tensor
        Output: [B, N_patches, Embedding_Dim]
        """
        x = (x.to(self.device) - self.mean) / (self.std + 1e-8)
        
        with torch.no_grad():
            # AnySat expects a dict with the modality key defined in the YAML
            embeddings = self.model.encode({"lidar_17ch": x})
        return embeddings

class GeoSightEngine:
    def __init__(self, stats_path="lidar_17ch_stats.json"):
        self.encoder = AnySatEncoder(stats_path=stats_path)

    def rasterize(self, las_path):
        """
        Week 1 Priority: Probabilistic Point Cloud (PPC) implementation.
        Extracts 17 channels from YellowScan VX LiDAR data.
        """
        with laspy.open(las_path) as fh:
            las = fh.read()
            
        # 0.5m GSD Binning
        x_bins = np.floor((las.x - las.header.x_min) / 0.5).astype(int)
        y_bins = np.floor((las.y - las.header.y_min) / 0.5).astype(int)
        
        grid_h, grid_w = x_bins.max() + 1, y_bins.max() + 1
        tensor = np.zeros((17, grid_h, grid_w))
        
        # Channel 0: Z-Mean (Surface Model)
        # Channel 1: Intensity Mean
        # Channel 2: Return Ratio (Penetration index)
        # Channels 3-12: Height Percentiles (10th to 100th)
        # Channel 13: Vertical Variance (Subsurface Proxy)
        # Channel 14: Point Density
        # Channel 15-16: Pulse Width / Echo Width (if available)
        
        tensor[0, x_bins, y_bins] = las.z
        tensor[1, x_bins, y_bins] = las.intensity
        
        # Subsurface Proxy: Vertical distribution variance
        # (Simplified for the MVP logic block)
        z_var = np.var(las.z) 
        tensor[13, x_bins, y_bins] = z_var
        
        return torch.from_numpy(tensor).float()

    def predict(self, las_path):
        """Main pipeline: Raster -> AnySat -> Signatures"""
        raster = self.rasterize(las_path).unsqueeze(0) # Add batch dim
        signatures = self.encoder(raster)
        return signatures
