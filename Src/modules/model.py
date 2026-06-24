import os
import time
import random
import json
import torch
import tqdm
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel
from .layers import MaskAttention, SelfAttentionFeatureExtract, MLP
from .fake_news_detector import FakeNewsDetector
from .dataset import SampledRationaleDataset
from Src.utils.utils import data2gpu, Averager, metrics, Recorder, get_monthly_path, get_tensorboard_writer, process_test_results, ensure_relative_path
from Src.utils.dataloader import get_dataloader


class CogniDirModel(nn.Module):
    """CogniDir model for fake news detection with cognitive malicious comments."""
    
    def __init__(self, config):
        super(CogniDirModel, self).__init__()
        self.rationale_number = config['rationale_number']
        self.emb_dim = config['emb_dim']
        
        bert_path = ensure_relative_path(config['bert_path'], 'bert_path')
        local_files_only = os.path.isdir(bert_path)
        
        # BERT encoders for content and rationales
        self.bert_content = BertModel.from_pretrained(
            bert_path,
            local_files_only=local_files_only
        ).requires_grad_(False)
        self.bert_FTR = BertModel.from_pretrained(
            bert_path,
            local_files_only=local_files_only
        ).requires_grad_(False)
        
        # Enable gradient for last BERT layer
        for name, param in self.bert_content.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
        for name, param in self.bert_FTR.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
        
        # Attention and MLP modules
        self.content_attention = MaskAttention(self.emb_dim)
        self.mlp = MLP(self.emb_dim, config['model']['mlp']['dims'], config['model']['mlp']['dropout'], output_layer=True)
        
        # Cross-attention modules for each rationale
        self.cross_attention_content = nn.ModuleList([
            SelfAttentionFeatureExtract(1, self.emb_dim) for _ in range(self.rationale_number)
        ])
        self.cross_attention_ftr = nn.ModuleList([
            SelfAttentionFeatureExtract(1, self.emb_dim) for _ in range(self.rationale_number)
        ])
        
        # Score mappers for rationale weighting
        self.score_mapper_ftr = nn.ModuleList([
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
        
        # Aggregation layers
        self.linear_Aggregate = nn.Linear(self.emb_dim * (self.rationale_number + 1), self.emb_dim)
        self.linear_concat = nn.Linear(self.emb_dim * 2, self.emb_dim)

    def forward(self, **kwargs):
        """Forward pass through the model."""
        content, content_masks = kwargs['content'], kwargs['content_masks']
        FTRs = [kwargs[f'rationale_{i+1}'] for i in range(self.rationale_number)]
        FTR_masks = [kwargs[f'rationale_{i+1}_masks'] for i in range(self.rationale_number)]
        
        # Encode content and rationales with BERT
        content_feature = self.bert_content(content, attention_mask=content_masks)[0]
        FTR_features = [self.bert_FTR(ftr, attention_mask=mask)[0] for ftr, mask in zip(FTRs, FTR_masks)]
        
        # Extract expert features from each rationale
        experts = []
        reweight_scores = []
        for i in range(self.rationale_number):
            mutual_content_FTR, _ = self.cross_attention_content[i](content_feature, FTR_features[i], content_masks)
            expert = torch.mean(mutual_content_FTR, dim=1)
            mutual_FTR_content, _ = self.cross_attention_ftr[i](FTR_features[i], content_feature, FTR_masks[i])
            mutual_FTR_content = torch.mean(mutual_FTR_content, dim=1)
            reweight_score = self.score_mapper_ftr[i](mutual_FTR_content)
            reweight_expert = reweight_score * expert
            experts.append(reweight_expert)
            reweight_scores.append(reweight_score)
        
        # Aggregate features and predict
        attn_content, _ = self.content_attention(content_feature, mask=content_masks)
        all_feature = torch.cat([attn_content] + experts, dim=-1)
        final_feature = self.linear_Aggregate(all_feature)
        label_pred = self.mlp(final_feature)
        gate_value = torch.cat(reweight_scores, dim=1)
        
        res = {
            'classify_pred': torch.sigmoid(label_pred.squeeze(1)),
            'gate_value': gate_value,
            'final_feature': final_feature,
            'content_feature': attn_content
        }
        for i in range(self.rationale_number):
            res[f'rationale_{i+1}_feature'] = experts[i]
        return res

def _extract_tensors_from_tensor_dataset(tensor_dataset):
    """Extract tensors from TensorDataset."""
    return list(tensor_dataset.tensors)

def build_selected_indices_for_train(num_samples, full_k=12, probs=(0.34, 0.33, 0.33)):
    """Build selected rationale indices for training based on group probabilities."""
    p_base, p_med, p_str = probs
    groups = {
        'original': [1, 2, 3],
        'base': [4, 5, 6],
        'medium': [7, 8, 9],
        'strong': [10, 11, 12]
    }
    selectable_groups = ['base', 'medium', 'strong']
    selected_indices_all = []
    for i in range(num_samples):
        fixed = [1, 4, 7, 10]
        selected = list(fixed)
        used_in_group = {'base': [4], 'medium': [7], 'strong': [10]}
        extra_needed = 2
        while extra_needed > 0:
            r = random.random()
            if r < p_base:
                cat = 'base'
            elif r < p_base + p_med:
                cat = 'medium'
            else:
                cat = 'strong'
            available = [x for x in groups[cat] if x not in used_in_group[cat]]
            if len(available) == 0:
                found = False
                for other in selectable_groups:
                    available2 = [x for x in groups[other] if x not in used_in_group[other]]
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
        if len(selected) < 6:
            for groupname in selectable_groups:
                for idx in groups[groupname]:
                    if idx not in selected:
                        selected.append(idx)
                        if len(selected) == 6:
                            break
                if len(selected) == 6:
                    break
        selected_indices_all.append(selected)
    return selected_indices_all

def build_dataset_from_raw_loader(raw_loader, selected_indices_all, batch_size, shuffle=False):
    raw_dataset = raw_loader.dataset
    raw_tensors = _extract_tensors_from_tensor_dataset(raw_dataset)
    sampled_ds = SampledRationaleDataset(raw_tensors, full_k=12, selected_indices_per_sample=selected_indices_all)
    new_loader = DataLoader(dataset=sampled_ds, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=False)
    return new_loader

def split_val_loader_into_three(val_loader, batch_size, shuffle=False):
    """Split validation loader into three groups: base, medium, strong."""
    raw_dataset = val_loader.dataset
    tensors = _extract_tensors_from_tensor_dataset(raw_dataset)
    num = tensors[0].shape[0]
    third = num // 3
    splits = {}
    names = ['base', 'medium', 'strong']
    for i, name in enumerate(names):
        start = i * third
        end = (i + 1) * third if i < 2 else num
        sliced = []
        for t in tensors:
            sliced.append(t[start:end])
        ds = TensorDataset(*sliced)
        loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=False)
        splits[name] = loader
    return splits

def compute_proportions(metrics: dict, beta: float = 5.0, eta: float = 0.1) -> tuple:
    """Compute sampling proportions using InfoDirichlet Resampling (IDR).
    
    Maps group-wise vulnerability scores to Dirichlet distribution parameters
    to determine next-round sampling weights.
    """
    eps = 1e-12
    scores = {}

    for cat in ['base', 'medium', 'strong']:
        m_loss = float(metrics[cat]['loss'])
        a = float(metrics[cat]['acc'])

        # q_j = exp(-m_loss)
        q = float(np.exp(-m_loss))

        # numerical clamp to avoid log(0)
        a = min(max(a, eps), 1.0 - eps)
        q = min(max(q, eps), 1.0 - eps)

        # s_j = Bernoulli cross-entropy from a_j to q_j
        s = -a * np.log(q) - (1.0 - a) * np.log(1.0 - q)
        scores[cat] = s

    alpha = {cat: float(np.exp(beta * scores[cat]) + eta) for cat in ['base', 'medium', 'strong']}
    total_alpha = sum(alpha.values())
    return tuple(alpha[cat] / total_alpha for cat in ['base', 'medium', 'strong'])


class Trainer:
    """Trainer class for CogniDir model with adaptive resampling."""
    
    def __init__(self, config, writer):
        self.config = config
        self.writer = writer
        self.rationale_number = config['rationale_number']
        
        # Check and create save path
        if self.config.get('save_param_dir') is None:
            raise ValueError("save_param_dir must be provided in config")
        save_param_dir = ensure_relative_path(self.config['save_param_dir'], 'save_param_dir')
        self.save_path = os.path.join(
            save_param_dir,
            f"{self.config['model_name']}_{self.config['data_name']}",
            str(self.config['month']))
        os.makedirs(self.save_path, exist_ok=True)
        self.sample_probs = tuple(config.get('initial_group_probs', (0.34, 0.33, 0.33)))
        self.tol = config.get('converge_tol', 1e-4)

    def train(self, logger=None):
        st_tm = time.time()
        self.model = FakeNewsDetector(self.config)
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        train_loader_raw = get_dataloader(
            get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'train.json'),
            self.config['max_len'], self.config['batchsize'], shuffle=True,
            bert_path=self.config['bert_path'], data_type=self.config['data_type'], 
            rationale_number=12
        )
        val_loader = get_dataloader(
            get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'val.json'),
            self.config['max_len'], self.config['batchsize'], shuffle=False,
            bert_path=self.config['bert_path'], data_type=self.config['data_type'], 
            rationale_number=self.rationale_number
        )
        test_loader = get_dataloader(
            get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'test.json'),
            self.config['max_len'], self.config['batchsize'], shuffle=False,
            bert_path=self.config['bert_path'], data_type=self.config['data_type'], 
            rationale_number=self.rationale_number
        )
        val_splits = split_val_loader_into_three(val_loader, batch_size=self.config['batchsize'], shuffle=False)
        raw_train_dataset = train_loader_raw.dataset
        raw_train_num = raw_train_dataset.tensors[0].shape[0]
        selected_indices_train = build_selected_indices_for_train(raw_train_num, full_k=12, probs=self.sample_probs)
        train_loader = build_dataset_from_raw_loader(train_loader_raw, selected_indices_train, batch_size=self.config['batchsize'], shuffle=True)
        groups = {'base': [4, 5, 6], 'medium': [7, 8, 9], 'strong': [10, 11, 12]}
        for epoch in range(self.config['epoch']):
            print(f'---------- epoch {epoch} ----------')
            self.model.train()
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss_classify = Averager()
            for batch in train_data_iter:
                batch_data = data2gpu(batch, self.config['use_cuda'], data_type=self.config['data_type'], rationale_number=self.rationale_number)
                label = batch_data['label']
                batch_input_data = {**self.config, **batch_data}
                res = self.model(**batch_input_data)
                loss_classify = loss_fn(res['classify_pred'], label.float())
                optimizer.zero_grad()
                loss_classify.backward()
                optimizer.step()
                avg_loss_classify.add(loss_classify.item())
            cur_train_loss = avg_loss_classify.item()
            print(f'epoch {epoch} train loss: {cur_train_loss}')
            converged = False
            if cur_train_loss < 0.5:
                converged = True
            if converged:
                print('train loss converged -> evaluate val splits and adjust sampling...')
                metrics = {}
                for cat in ['base', 'medium', 'strong']:
                    results, aux = self.test(val_splits[cat])
                    val_loss = aux['val_avg_loss_classify'].item()
                    val_acc = results['acc']
                    metrics[cat] = {'loss': val_loss, 'acc': val_acc}
                    print(f'val {cat} loss: {val_loss}, acc: {val_acc}')
                # Compute new proportions using IDR
                self.sample_probs = compute_proportions(metrics)
                
                # Allocate samples based on proportions
                categories = ['base', 'medium', 'strong']
                allocations = [{'cat': cat, 'prob': p, 'count': int(np.ceil(2 * p))} for cat, p in zip(categories, self.sample_probs)]
                allocations.sort(key=lambda x: x['prob'], reverse=True)
                total_count = sum(a['count'] for a in allocations)
                if total_count > 2:
                    for a in allocations[::-1]:
                        if total_count > 2 and a['count'] > 0:
                            a['count'] -= 1
                            total_count -= 1
                
                # Rebuild train loader with new sampling
                selected_indices_train = []
                for _ in range(raw_train_num):
                    fixed = [1, 4, 7, 10]
                    selected = list(fixed)
                    used_in_group = {'base': [4], 'medium': [7], 'strong': [10]}
                    for alloc in allocations:
                        cat = alloc['cat']
                        count = alloc['count']
                        available = [x for x in groups[cat] if x not in used_in_group[cat]]
                        for _ in range(count):
                            if available:
                                sel_idx = available[0]
                                selected.append(sel_idx)
                                used_in_group[cat].append(sel_idx)
                                available.pop(0)
                    # Fill if needed
                    if len(selected) < 6:
                        for cat in categories:
                            available = [x for x in groups[cat] if x not in used_in_group[cat]]
                            for idx in available:
                                selected.append(idx)
                                used_in_group[cat].append(idx)
                                if len(selected) == 6:
                                    break
                            if len(selected) == 6:
                                break
                    selected_indices_train.append(selected)
                train_loader = build_dataset_from_raw_loader(train_loader_raw, selected_indices_train, batch_size=self.config['batchsize'], shuffle=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'parameter_bert.pkl'))
        results, label, pred, ae, accuracy = self.predict(test_loader)
        test_dir = os.path.join('logs', 'test', self.config['model_name'] + '_' + self.config['data_name'])
        os.makedirs(test_dir, exist_ok=True)
        test_res_path = os.path.join(test_dir, 'month_' + str(self.config['month']) + '.json')
        process_test_results(
            get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'test.json'),
            test_res_path, label, pred, None, ae, accuracy
        )
        if self.writer is not None:
            self.writer.add_scalars(f'month_{self.config["month"]}/test', results)
        print('test results:', results)
        return results, os.path.join(self.save_path, 'parameter_bert.pkl'), epoch

    def test(self, dataloader):
        loss_fn = torch.nn.BCELoss()
        pred, label = [], []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        avg_loss_classify = Averager()
        for batch in data_iter:
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'], data_type=self.config['data_type'], rationale_number=self.rationale_number)
                batch_label = batch_data['label']
                batch_input_data = {**self.config, **batch_data}
                res = self.model(**batch_input_data)
                loss_classify = loss_fn(res['classify_pred'], batch_label.float())
                label.extend(batch_label.cpu().numpy().tolist())
                pred.extend(res['classify_pred'].cpu().numpy().tolist())
                avg_loss_classify.add(loss_classify.item())
        return metrics(label, pred), {'val_avg_loss_classify': avg_loss_classify, 'pred': pred, 'label': label}

    def predict(self, dataloader):
        if self.config.get('eval_mode', False):
            if 'eval_model_path' not in self.config or self.config['eval_model_path'] is None:
                raise ValueError("eval_model_path must be provided when eval_mode is True")
            self.model = FakeNewsDetector(self.config)
            if self.config['use_cuda']:
                self.model = self.model.cuda()
            print('========== in test process ==========')
            print('now load in test model...')
            eval_model_path = ensure_relative_path(self.config['eval_model_path'], 'eval_model_path')
            self.model.load_state_dict(torch.load(eval_model_path))
        pred, label, ae, accuracy = [], [], [], []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for batch in data_iter:
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'], data_type=self.config['data_type'], rationale_number=self.rationale_number)
                batch_label = batch_data['label']
                batch_input_data = {**self.config, **batch_data}
                res = self.model(**batch_input_data)
                batch_pred = res['classify_pred']
                cur_labels = batch_label.cpu().numpy().tolist()
                cur_preds = batch_pred.cpu().numpy().tolist()
                label.extend(cur_labels)
                pred.extend(cur_preds)
                ae_list = [abs(p - l) for p, l in zip(cur_preds, cur_labels)]
                accuracy_list = [1 if e < 0.5 else 0 for e in ae_list]
                ae.extend(ae_list)
                accuracy.extend(accuracy_list)
        return metrics(label, pred), label, pred, ae, accuracy