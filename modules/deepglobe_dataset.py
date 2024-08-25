import pandas

import torch
from torch.utils.data import Dataset

from PIL import Image
import os

class DeepGlobeDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.root_dir = 'data/deepglobe/'
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sat_image = Image.open(os.path.join(self.root_dir, self.metadata.iloc[idx].sat_image_path))
        mask = Image.open(os.path.join(self.root_dir, self.metadata.iloc[idx].mask_path)).convert('L')

        if self.transform:
            crops, masks = self.transform((sat_image, mask))
            
        crops_tensor = torch.stack(crops)
        masks_tensor = torch.stack(masks)
        
        return crops_tensor, masks_tensor