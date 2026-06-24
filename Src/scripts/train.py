"""
Training Script for CogniDir

This script implements the training process with InfoDirichlet Resampling,
including adaptive resampling based on group-wise vulnerability scores.
"""

import os
import sys
import argparse
import json
import time
import random
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
base_dir = os.path.dirname(__file__)
project_root = os.path.normpath(os.path.join(base_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Src.utils.utils import setup_project_cwd
setup_project_cwd()

from Src.modules.fake_news_detector import FakeNewsDetector
from Src.modules.initialized_proportion import InitializedProportion
from Src.modules.infodirichlet_resampling import InfoDirichletResampling
from Src.utils.dataloader import get_dataloader
from Src.utils.utils import (
    data2gpu, Averager, metrics, get_monthly_path, ensure_relative_path
)


def split_val_loader_into_three(val_loader, batch_size, shuffle=False):
    """Split validation loader into three attack groups."""
    raw_dataset = val_loader.dataset
    tensors = list(raw_dataset.tensors)
    num = tensors[0].shape[0]
    third = num // 3
    splits = {}
    names = ['base', 'medium', 'strong']
    
    for i, name in enumerate(names):
        start = i * third
        end = (i + 1) * third if i < 2 else num
        sliced = [t[start:end] for t in tensors]
        ds = TensorDataset(*sliced)
        loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, 
                          num_workers=1, pin_memory=False)
        splits[name] = loader
    return splits


def build_dataset_from_indices(raw_loader, selected_indices_all, batch_size, shuffle=False):
    """Build dataset from selected rationale indices."""
    from Src.modules.dataset import SampledRationaleDataset
    from Src.modules.model import _extract_tensors_from_tensor_dataset
    
    raw_dataset = raw_loader.dataset
    raw_tensors = _extract_tensors_from_tensor_dataset(raw_dataset)
    sampled_ds = SampledRationaleDataset(
        raw_tensors, full_k=12, selected_indices_per_sample=selected_indices_all
    )
    new_loader = DataLoader(
        dataset=sampled_ds, batch_size=batch_size, shuffle=shuffle, 
        num_workers=1, pin_memory=False
    )
    return new_loader


class Trainer:
    """Trainer for CogniDir with adaptive resampling."""
    
    def __init__(self, config, writer):
        self.config = config
        self.writer = writer
        self.rationale_number = config['rationale_number']
        
        # Initialize modules
        self.initialized_proportion = InitializedProportion(
            initial_probs=tuple(config.get('initial_group_probs', (0.34, 0.33, 0.33)))
        )
        self.idr = InfoDirichletResampling(
            beta=config.get('idr_beta', 5.0),
            eta=config.get('idr_eta', 0.1)
        )
        
        self.sample_probs = self.initialized_proportion.p_base, \
                           self.initialized_proportion.p_medium, \
                           self.initialized_proportion.p_strong
        
        # Check and create save path
        if self.config.get('save_param_dir') is None:
            raise ValueError("save_param_dir must be provided in config")
        save_param_dir = ensure_relative_path(self.config['save_param_dir'], 'save_param_dir')
        self.save_path = os.path.join(
            save_param_dir,
            f"{self.config['model_name']}_{self.config['data_name']}",
            str(self.config['month'])
        )
        os.makedirs(self.save_path, exist_ok=True)
    
    def train(self, logger=None):
        """Main training loop with adaptive resampling."""
        # Initialize model
        self.model = FakeNewsDetector(self.config)
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), 
            lr=self.config['lr'], 
            weight_decay=self.config['weight_decay']
        )
        
        # Load data
        train_loader_raw = get_dataloader(
            get_monthly_path(self.config['data_type'], self.config['root_path'], 
                           self.config['month'], 'train.json'),
            self.config['max_len'], self.config['batchsize'], shuffle=True,
            bert_path=self.config['bert_path'], data_type=self.config['data_type'], 
            rationale_number=12
        )
        val_loader = get_dataloader(
            get_monthly_path(self.config['data_type'], self.config['root_path'], 
                           self.config['month'], 'val.json'),
            self.config['max_len'], self.config['batchsize'], shuffle=False,
            bert_path=self.config['bert_path'], data_type=self.config['data_type'], 
            rationale_number=self.rationale_number
        )
        test_loader = get_dataloader(
            get_monthly_path(self.config['data_type'], self.config['root_path'], 
                           self.config['month'], 'test.json'),
            self.config['max_len'], self.config['batchsize'], shuffle=False,
            bert_path=self.config['bert_path'], data_type=self.config['data_type'], 
            rationale_number=self.rationale_number
        )
        
        # Split validation into groups
        val_splits = split_val_loader_into_three(
            val_loader, batch_size=self.config['batchsize'], shuffle=False
        )
        
        # Initialize training data sampling
        raw_train_dataset = train_loader_raw.dataset
        raw_train_num = raw_train_dataset.tensors[0].shape[0]
        selected_indices_train = self.initialized_proportion.build_selected_indices(
            raw_train_num, full_k=12
        )
        train_loader = build_dataset_from_indices(
            train_loader_raw, selected_indices_train, 
            batch_size=self.config['batchsize'], shuffle=True
        )
        
        groups = {'base': [4, 5, 6], 'medium': [7, 8, 9], 'strong': [10, 11, 12]}
        
        # Training loop
        for epoch in range(self.config['epoch']):
            print(f'---------- epoch {epoch} ----------')
            self.model.train()
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss_classify = Averager()
            
            for batch in train_data_iter:
                batch_data = data2gpu(
                    batch, self.config['use_cuda'], 
                    data_type=self.config['data_type'], 
                    rationale_number=self.rationale_number
                )
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
            
            # Adaptive resampling when loss converges
            if cur_train_loss < 0.5:
                print('train loss converged -> evaluate val splits and adjust sampling...')
                group_metrics = {}
                
                for cat in ['base', 'medium', 'strong']:
                    results, aux = self.test(val_splits[cat])
                    val_loss = aux['val_avg_loss_classify'].item()
                    val_acc = results['acc']
                    group_metrics[cat] = {'loss': val_loss, 'acc': val_acc}
                    print(f'val {cat} loss: {val_loss}, acc: {val_acc}')
                
                # Compute new proportions using IDR
                self.sample_probs = self.idr.compute_proportions(group_metrics)
                allocations = self.idr.allocate_samples(self.sample_probs, total_extra=2)
                
                # Rebuild training data with new sampling
                selected_indices_train = []
                for _ in range(raw_train_num):
                    fixed = [1, 4, 7, 10]
                    selected = list(fixed)
                    used_in_group = {'base': [4], 'medium': [7], 'strong': [10]}
                    
                    for cat, count in allocations.items():
                        available = [x for x in groups[cat] if x not in used_in_group[cat]]
                        for _ in range(count):
                            if available:
                                sel_idx = available[0]
                                selected.append(sel_idx)
                                used_in_group[cat].append(sel_idx)
                                available.pop(0)
                    
                    # Fill to ensure at least 6 rationales
                    if len(selected) < 6:
                        for cat in ['base', 'medium', 'strong']:
                            available = [x for x in groups[cat] if x not in used_in_group[cat]]
                            for idx in available:
                                selected.append(idx)
                                if len(selected) == 6:
                                    break
                            if len(selected) == 6:
                                break
                    
                    selected_indices_train.append(selected)
                
                train_loader = build_dataset_from_indices(
                    train_loader_raw, selected_indices_train, 
                    batch_size=self.config['batchsize'], shuffle=True
                )
        
        # Save model
        model_path = os.path.join(self.save_path, 'parameter_bert.pkl')
        torch.save(self.model.state_dict(), model_path)
        
        # Evaluate on test set
        results, label, pred, ae, accuracy = self.predict(test_loader)
        print('Test results:', results)
        
        return results, model_path, epoch
    
    def test(self, dataloader):
        """Evaluate on validation set."""
        loss_fn = torch.nn.BCELoss()
        pred, label = [], []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        avg_loss_classify = Averager()
        
        for batch in data_iter:
            with torch.no_grad():
                batch_data = data2gpu(
                    batch, self.config['use_cuda'], 
                    data_type=self.config['data_type'], 
                    rationale_number=self.rationale_number
                )
                batch_label = batch_data['label']
                batch_input_data = {**self.config, **batch_data}
                res = self.model(**batch_input_data)
                loss_classify = loss_fn(res['classify_pred'], batch_label.float())
                
                label.extend(batch_label.cpu().numpy().tolist())
                pred.extend(res['classify_pred'].cpu().numpy().tolist())
                avg_loss_classify.add(loss_classify.item())
        
        return metrics(label, pred), {
            'val_avg_loss_classify': avg_loss_classify, 
            'pred': pred, 
            'label': label
        }
    
    def predict(self, dataloader):
        """Predict on test set."""
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
                batch_data = data2gpu(
                    batch, self.config['use_cuda'], 
                    data_type=self.config['data_type'], 
                    rationale_number=self.rationale_number
                )
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


def main():
    parser = argparse.ArgumentParser(description='Train CogniDir model')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='CogniDir')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--max_len', type=int, default=5)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    
    # Data parameters
    parser.add_argument('--root_path', type=str, default='Data/Processed/rumour')
    parser.add_argument('--data_name', type=str, default='rumour')
    parser.add_argument('--data_type', type=str, default='rationale')
    parser.add_argument('--bert_path', type=str, default='BERT')
    
    # IDR parameters
    parser.add_argument('--idr_beta', type=float, default=5.0)
    parser.add_argument('--idr_eta', type=float, default=0.1)
    parser.add_argument('--initial_group_probs', type=float, nargs=3, 
                       default=[0.34, 0.33, 0.33])
    
    # System parameters
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=3759)
    
    # Output directories
    parser.add_argument('--save_param_dir', type=str, default='param_model')
    parser.add_argument('--tensorboard_dir', type=str, default=None)
    
    # Evaluation parameters
    parser.add_argument('--eval_mode', type=bool, default=False)
    parser.add_argument('--eval_model_path', type=str, default=None,
                       help='Relative path to model checkpoint for evaluation mode')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Build config
    config = {
        'use_cuda': True,
        'seed': args.seed,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'weight_decay': args.weight_decay,
        'rationale_number': 6,
        'model': {
            'mlp': {'dims': [384], 'dropout': 0.2}
        },
        'emb_dim': args.emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'root_path': args.root_path,
        'data_type': args.data_type,
        'data_name': args.data_name,
        'bert_path': args.bert_path,
        'idr_beta': args.idr_beta,
        'idr_eta': args.idr_eta,
        'initial_group_probs': args.initial_group_probs,
        'month': 1,
        'save_param_dir': args.save_param_dir,
        'tensorboard_dir': args.tensorboard_dir,
        'eval_mode': args.eval_mode,
        'eval_model_path': args.eval_model_path
    }
    
    # Initialize trainer
    trainer = Trainer(config, None)
    
    # Train
    results, model_path, epoch = trainer.train()
    
    print(f'Training completed. Final epoch: {epoch}')
    print(f'Model saved to: {model_path}')
    print(f'Test results: {results}')


if __name__ == '__main__':
    main()
