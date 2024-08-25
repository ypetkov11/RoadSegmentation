import torch

def collate_fn(batch):
    images = [img for crops, masks in batch for img in crops]
    masks = [msk for crops, masks in batch for msk in masks]
    return torch.stack(images), torch.stack(masks)