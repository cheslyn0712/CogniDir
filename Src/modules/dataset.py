"""
Dataset Module

This module provides dataset classes for sampling selected rationales
during training with adaptive resampling.
"""

import torch
from torch.utils.data import Dataset


class SampledRationaleDataset(Dataset):
    """Dataset for sampling selected rationales during training."""
    
    def __init__(self, raw_tensors, full_k=12, selected_indices_per_sample=None):
        """
        Initialize dataset with selected rationale indices.
        
        Args:
            raw_tensors: List of tensors from raw dataset
            full_k: Total number of rationales (default: 12)
            selected_indices_per_sample: List of selected indices for each sample
        """
        self.raw_tensors = raw_tensors
        self.full_k = full_k
        self.selected_indices_per_sample = selected_indices_per_sample
        self.num_samples = raw_tensors[0].shape[0]
        
        # Index mapping
        self.content_idx = 0
        self.content_mask_idx = 1
        self.rationale_token_start = 2
        self.rationale_mask_start = 2 + full_k
        self.label_idx = 2 + full_k * 2
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get item with selected rationales."""
        content = self.raw_tensors[self.content_idx][idx]
        content_mask = self.raw_tensors[self.content_mask_idx][idx]
        label = self.raw_tensors[self.label_idx][idx]
        
        if self.selected_indices_per_sample is None:
            raise ValueError("selected_indices_per_sample must be provided")
        
        sel = self.selected_indices_per_sample[idx]
        rationale_tensors = []
        rationale_mask_tensors = []
        
        for r_idx in sel:
            token_tensor = self.raw_tensors[self.rationale_token_start + (r_idx - 1)][idx]
            mask_tensor = self.raw_tensors[self.rationale_mask_start + (r_idx - 1)][idx]
            rationale_tensors.append(token_tensor)
            rationale_mask_tensors.append(mask_tensor)
        
        return_tuple = (content, content_mask, *rationale_tensors, *rationale_mask_tensors, label)
        return return_tuple
