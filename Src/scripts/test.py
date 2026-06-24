"""
Test Script for CogniDir

This script evaluates the trained CogniDir model on test datasets.
"""

import os
import sys
import argparse
import torch
import numpy as np
import random
import tqdm

# Add project root to path
base_dir = os.path.dirname(__file__)
project_root = os.path.normpath(os.path.join(base_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Src.utils.utils import setup_project_cwd, ensure_relative_path
setup_project_cwd()

from Src.modules.fake_news_detector import FakeNewsDetector
from Src.utils.dataloader import get_dataloader
from Src.utils.utils import data2gpu, metrics, get_monthly_path


class Tester:
    """Tester for CogniDir model evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.rationale_number = config['rationale_number']
        
        # Initialize model
        self.model = FakeNewsDetector(self.config)
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        
        # Load model weights
        if config.get('model_path'):
            model_path = ensure_relative_path(config['model_path'], 'model_path')
            print(f'Loading model from {model_path}')
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise ValueError("model_path must be provided for testing")
    
    def test(self, test_loader):
        """Evaluate on test set."""
        pred, label, ae, accuracy = [], [], [], []
        self.model.eval()
        data_iter = tqdm.tqdm(test_loader)
        
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
        
        # Compute metrics
        test_metrics = metrics(label, pred)
        
        return test_metrics, label, pred, ae, accuracy
    
    def evaluate_group_wise(self, test_loaders):
        """Evaluate on different attack groups."""
        results = {}
        
        for group_name, loader in test_loaders.items():
            print(f'Evaluating {group_name} group...')
            metrics_dict, label, pred, ae, accuracy = self.test(loader)
            results[group_name] = {
                'metrics': metrics_dict,
                'label': label,
                'pred': pred
            }
            print(f'{group_name} - F1: {metrics_dict["metric"]:.4f}, '
                  f'Acc: {metrics_dict["acc"]:.4f}')
        
        return results


def split_test_loader_into_groups(test_loader, batch_size):
    """Split test loader into attack groups."""
    from torch.utils.data import DataLoader, TensorDataset
    
    raw_dataset = test_loader.dataset
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
        loader = DataLoader(
            dataset=ds, batch_size=batch_size, shuffle=False, 
            num_workers=1, pin_memory=False
        )
        splits[name] = loader
    return splits


def main():
    parser = argparse.ArgumentParser(description='Test CogniDir model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Relative path to trained model checkpoint (e.g. param_model/CogniDir_rumour/1/parameter_bert.pkl)')
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--max_len', type=int, default=5)
    parser.add_argument('--batchsize', type=int, default=64)
    
    # Data parameters
    parser.add_argument('--root_path', type=str, default='Data/Processed/rumour')
    parser.add_argument('--data_name', type=str, default='rumour')
    parser.add_argument('--data_type', type=str, default='rationale')
    parser.add_argument('--bert_path', type=str, default='BERT')
    
    # Output parameters removed - no saving
    
    # System parameters
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=3759)
    parser.add_argument('--group_wise', action='store_true',
                       help='Evaluate on different attack groups separately')
    
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
        'rationale_number': 6,
        'model': {
            'mlp': {'dims': [384], 'dropout': 0.2}
        },
        'emb_dim': args.emb_dim,
        'root_path': args.root_path,
        'data_type': args.data_type,
        'data_name': args.data_name,
        'bert_path': args.bert_path,
        'model_path': args.model_path,
        'month': 1
    }
    
    # Load test data
    test_loader = get_dataloader(
        get_monthly_path(config['data_type'], config['root_path'], 
                        config['month'], 'test.json'),
        config['max_len'], config['batchsize'], shuffle=False,
        bert_path=config['bert_path'], 
        data_type=config['data_type'], 
        rationale_number=config['rationale_number']
    )
    
    # Initialize tester
    tester = Tester(config)
    
    if args.group_wise:
        # Evaluate on different attack groups
        test_splits = split_test_loader_into_groups(test_loader, config['batchsize'])
        results = tester.evaluate_group_wise(test_splits)
        
        print('\n=== Group-wise Results ===')
        for group_name, result in results.items():
            m = result['metrics']
            print(f'{group_name}: F1={m["metric"]:.4f}, '
                  f'Acc={m["acc"]:.4f}, '
                  f'Precision={m["precision"]:.4f}, '
                  f'Recall={m["recall"]:.4f}')
    else:
        # Evaluate on full test set
        print('Evaluating on test set...')
        test_metrics, label, pred, ae, accuracy = tester.test(test_loader)
        
        print('\n=== Test Results ===')
        print(f'F1 Score: {test_metrics["metric"]:.4f}')
        print(f'Accuracy: {test_metrics["acc"]:.4f}')
        print(f'Precision: {test_metrics["precision"]:.4f}')
        print(f'Recall: {test_metrics["recall"]:.4f}')
        print(f'AUC: {test_metrics["auc"]:.4f}')
        
        # Results are printed only, not saved


if __name__ == '__main__':
    main()
