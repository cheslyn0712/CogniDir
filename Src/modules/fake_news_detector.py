"""
Fake News Detector Module

This module implements the core fake news detection model with multi-expert
architecture, including content encoder, rationale encoders, cross-attention
modules, expert aggregation, and classification head.
"""

import torch
import torch.nn as nn
from transformers import BertModel
from .layers import MaskAttention, SelfAttentionFeatureExtract, MLP
from Src.utils.utils import ensure_relative_path
import os


class FakeNewsDetector(nn.Module):
    """
    Fake News Detector with multi-expert architecture.
    
    Components:
    - Content Encoder: BERT-based encoder for news content
    - Rationale Encoders: Multiple BERT encoders for adversarial comments
    - Cross-Attention Modules: Bidirectional attention between content and rationales
    - Expert Aggregation: Weighted aggregation of expert features
    - Classification Head: MLP for binary classification
    """
    
    def __init__(self, config):
        super(FakeNewsDetector, self).__init__()
        self.rationale_number = config['rationale_number']
        self.emb_dim = config['emb_dim']
        
        bert_path = ensure_relative_path(config['bert_path'], 'bert_path')
        local_files_only = os.path.isdir(bert_path)

        # Content Encoder: BERT for news content
        self.bert_content = BertModel.from_pretrained(
            bert_path, local_files_only=local_files_only
        ).requires_grad_(False)
        for name, param in self.bert_content.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True

        # Rationale Encoders: BERT for adversarial comments
        self.bert_rationale = BertModel.from_pretrained(
            bert_path, local_files_only=local_files_only
        ).requires_grad_(False)
        for name, param in self.bert_rationale.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
        
        # Content Attention: Extract important content features
        self.content_attention = MaskAttention(self.emb_dim)
        
        # Cross-Attention Modules: Bidirectional attention for each rationale
        self.cross_attention_content = nn.ModuleList([
            SelfAttentionFeatureExtract(1, self.emb_dim) 
            for _ in range(self.rationale_number)
        ])
        self.cross_attention_rationale = nn.ModuleList([
            SelfAttentionFeatureExtract(1, self.emb_dim) 
            for _ in range(self.rationale_number)
        ])
        
        # Score Mappers: Compute importance weights for each rationale
        self.score_mapper = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.emb_dim, config['model']['mlp']['dims'][0]),
                nn.BatchNorm1d(config['model']['mlp']['dims'][0]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config['model']['mlp']['dims'][0], 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(self.rationale_number)
        ])
        
        # Expert Aggregation: Aggregate content and expert features
        self.linear_aggregate = nn.Linear(
            self.emb_dim * (self.rationale_number + 1), 
            self.emb_dim
        )
        
        # Classification Head: MLP for final prediction
        self.mlp = MLP(
            self.emb_dim, 
            config['model']['mlp']['dims'], 
            config['model']['mlp']['dropout'], 
            output_layer=True
        )
    
    def forward(self, **kwargs):
        """
        Forward pass through the detector.
        
        Args:
            content: News content tokens
            content_masks: Content attention masks
            rationale_{i}: Rationale tokens for i-th rationale
            rationale_{i}_masks: Rationale attention masks
            
        Returns:
            Dictionary containing predictions and intermediate features
        """
        content = kwargs['content']
        content_masks = kwargs['content_masks']
        rationales = [kwargs[f'rationale_{i+1}'] for i in range(self.rationale_number)]
        rationale_masks = [kwargs[f'rationale_{i+1}_masks'] for i in range(self.rationale_number)]
        
        # Encode content and rationales
        content_features = self.bert_content(content, attention_mask=content_masks)[0]
        rationale_features = [
            self.bert_rationale(r, attention_mask=m)[0] 
            for r, m in zip(rationales, rationale_masks)
        ]
        
        # Extract expert features from each rationale
        experts = []
        reweight_scores = []
        
        for i in range(self.rationale_number):
            # Cross-attention: content -> rationale
            mutual_content_rationale, _ = self.cross_attention_content[i](
                content_features, rationale_features[i], content_masks
            )
            expert = torch.mean(mutual_content_rationale, dim=1)
            
            # Cross-attention: rationale -> content
            mutual_rationale_content, _ = self.cross_attention_rationale[i](
                rationale_features[i], content_features, rationale_masks[i]
            )
            mutual_rationale_content = torch.mean(mutual_rationale_content, dim=1)
            
            # Compute importance weight
            reweight_score = self.score_mapper[i](mutual_rationale_content)
            reweight_expert = reweight_score * expert
            
            experts.append(reweight_expert)
            reweight_scores.append(reweight_score)
        
        # Aggregate content and expert features
        attn_content, _ = self.content_attention(content_features, mask=content_masks)
        all_features = torch.cat([attn_content] + experts, dim=-1)
        final_feature = self.linear_aggregate(all_features)
        
        # Classification
        label_pred = self.mlp(final_feature)
        gate_value = torch.cat(reweight_scores, dim=1)
        
        res = {
            'classify_pred': torch.sigmoid(label_pred.squeeze(1)),
            'gate_value': gate_value,
            'final_feature': final_feature,
            'content_feature': attn_content
        }
        
        # Add individual expert features
        for i in range(self.rationale_number):
            res[f'rationale_{i+1}_feature'] = experts[i]
        
        return res
