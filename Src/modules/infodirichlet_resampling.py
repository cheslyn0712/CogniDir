"""
InfoDirichlet Resampling (IDR) Module

This module implements the InfoDirichlet Resampling mechanism that dynamically
adjusts training data proportions based on group-wise vulnerability scores.
It maps vulnerability scores to Dirichlet distribution parameters to determine
sampling weights for different attack groups.
"""

import numpy as np
from typing import Dict, Tuple


class InfoDirichletResampling:
    """
    InfoDirichlet Resampling mechanism for adaptive training.
    
    Computes sampling proportions by:
    1. Evaluating group-wise vulnerability scores
    2. Mapping scores to Dirichlet distribution parameters
    3. Computing normalized sampling weights
    """
    
    def __init__(self, beta: float = 5.0, eta: float = 0.1):
        """
        Initialize IDR parameters.
        
        Args:
            beta: Scaling factor for vulnerability scores (default: 5.0)
            eta: Smoothing parameter for Dirichlet (default: 0.1)
        """
        self.beta = beta
        self.eta = eta
        self.eps = 1e-12
    
    def compute_proportions(self, metrics: Dict[str, Dict[str, float]]) -> Tuple[float, float, float]:
        """
        Compute sampling proportions using InfoDirichlet Resampling.
        
        Vulnerability score computation:
        - a_j = accuracy for group j
        - q_j = exp(-loss_j)
        - s_j = -a_j * log(q_j) - (1-a_j) * log(1-q_j)
        
        Dirichlet parameter mapping:
        - alpha_j = exp(beta * s_j) + eta
        - p_j = alpha_j / sum_k alpha_k
        
        Args:
            metrics: Dictionary with keys 'base', 'medium', 'strong', each containing:
                - 'loss': Average cross-entropy loss
                - 'acc': Accuracy score
                
        Returns:
            Tuple of (p_base, p_medium, p_strong) sampling proportions
        """
        scores = {}
        
        for cat in ['base', 'medium', 'strong']:
            m_loss = float(metrics[cat]['loss'])
            a = float(metrics[cat]['acc'])
            
            # Compute q_j = exp(-loss)
            q = float(np.exp(-m_loss))
            
            # Numerical stability: clamp values
            a = np.clip(a, self.eps, 1.0 - self.eps)
            q = np.clip(q, self.eps, 1.0 - self.eps)
            
            # Compute vulnerability score: Bernoulli cross-entropy
            s = -a * np.log(q) - (1.0 - a) * np.log(1.0 - q)
            scores[cat] = s
        
        # Map to Dirichlet parameters
        alpha = {
            cat: float(np.exp(self.beta * scores[cat]) + self.eta) 
            for cat in ['base', 'medium', 'strong']
        }
        
        # Normalize to get proportions
        total_alpha = sum(alpha.values())
        proportions = tuple(alpha[cat] / total_alpha for cat in ['base', 'medium', 'strong'])
        
        return proportions
    
    def allocate_samples(self, proportions: Tuple[float, float, float], total_extra: int = 2) -> Dict[str, int]:
        """
        Allocate sample counts based on proportions.
        
        Args:
            proportions: Tuple of (p_base, p_medium, p_strong)
            total_extra: Total number of extra samples to allocate
            
        Returns:
            Dictionary mapping group names to sample counts
        """
        categories = ['base', 'medium', 'strong']
        allocations = [
            {'cat': cat, 'prob': p, 'count': int(np.ceil(total_extra * p))} 
            for cat, p in zip(categories, proportions)
        ]
        
        # Sort by probability (descending)
        allocations.sort(key=lambda x: x['prob'], reverse=True)
        
        # Adjust if total exceeds target
        total_count = sum(a['count'] for a in allocations)
        if total_count > total_extra:
            for a in allocations[::-1]:
                if total_count > total_extra and a['count'] > 0:
                    a['count'] -= 1
                    total_count -= 1
        
        return {a['cat']: a['count'] for a in allocations}
