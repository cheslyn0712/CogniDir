"""
Initialized Proportion Module

This module handles the initialization of sampling proportions for different
attack groups (base, medium, strong) during training. It implements the
initial sampling strategy based on group probabilities.
"""

import random
from typing import List, Tuple, Dict


class InitializedProportion:
    """Initialize sampling proportions for rationale groups."""
    
    def __init__(self, initial_probs: Tuple[float, float, float] = (0.34, 0.33, 0.33)):
        """
        Initialize with group probabilities.
        
        Args:
            initial_probs: Tuple of (p_base, p_medium, p_strong) probabilities
        """
        self.p_base, self.p_medium, self.p_strong = initial_probs
        
        # Define rationale groups
        self.groups = {
            'original': [1, 2, 3],
            'base': [4, 5, 6],
            'medium': [7, 8, 9],
            'strong': [10, 11, 12]
        }
        self.selectable_groups = ['base', 'medium', 'strong']
    
    def build_selected_indices(self, num_samples: int, full_k: int = 12) -> List[List[int]]:
        """
        Build selected rationale indices for training based on group probabilities.
        
        Args:
            num_samples: Number of training samples
            full_k: Total number of rationales (default: 12)
            
        Returns:
            List of selected rationale indices for each sample
        """
        selected_indices_all = []
        
        for i in range(num_samples):
            # Fixed indices: one from each group
            fixed = [1, 4, 7, 10]
            selected = list(fixed)
            used_in_group = {'base': [4], 'medium': [7], 'strong': [10]}
            extra_needed = 2
            
            # Sample additional rationales based on probabilities
            while extra_needed > 0:
                r = random.random()
                if r < self.p_base:
                    cat = 'base'
                elif r < self.p_base + self.p_medium:
                    cat = 'medium'
                else:
                    cat = 'strong'
                
                available = [x for x in self.groups[cat] if x not in used_in_group[cat]]
                if len(available) == 0:
                    # Try other groups if current group is exhausted
                    found = False
                    for other in self.selectable_groups:
                        available2 = [x for x in self.groups[other] if x not in used_in_group[other]]
                        if len(available2) > 0:
                            sel_idx = available2[0]
                            used_in_group[other].append(sel_idx)
                            selected.append(sel_idx)
                            extra_needed -= 1
                            found = True
                            break
                    if not found:
                        break
                else:
                    sel_idx = available[0]
                    used_in_group[cat].append(sel_idx)
                    selected.append(sel_idx)
                    extra_needed -= 1
            
            # Fill to ensure at least 6 rationales per sample
            if len(selected) < 6:
                for groupname in self.selectable_groups:
                    for idx in self.groups[groupname]:
                        if idx not in selected:
                            selected.append(idx)
                            if len(selected) == 6:
                                break
                    if len(selected) == 6:
                        break
            
            selected_indices_all.append(selected)
        
        return selected_indices_all
