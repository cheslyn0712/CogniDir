from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
from datetime import datetime as dt
from tensorboardX import SummaryWriter
import os
import json
import pandas as pd
import torch

def get_project_root():
    """Return project root directory for resolving relative paths."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def ensure_relative_path(path, name='path'):
    """Ensure a path is relative; absolute paths are not allowed."""
    if os.path.isabs(path):
        raise ValueError(f'{name} must be a relative path, got: {path}')
    return os.path.normpath(path)


def join_relative_path(*parts):
    """Join and normalize path parts as a relative path."""
    return os.path.normpath(os.path.join(*parts))


def setup_project_cwd():
    """Switch working directory to project root so relative paths resolve correctly."""
    os.chdir(get_project_root())

class Recorder:
    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("current", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def metrics(y_true, y_pred):
    all_metrics = {}
    try:
        all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        all_metrics['auc'] = -1
    try:
        all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    except ValueError:
        all_metrics['spauc'] = -1
    y_pred = np.around(np.array(y_pred)).astype(int)
    all_metrics['metric'] = f1_score(y_true, y_pred, average='macro')
    try:
        all_metrics['f1_real'], all_metrics['f1_fake'] = f1_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['f1_real'], all_metrics['f1_fake'] = -1, -1
    all_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    try:
        all_metrics['recall_real'], all_metrics['recall_fake'] = recall_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['recall_real'], all_metrics['recall_fake'] = -1, -1
    all_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    try:
        all_metrics['precision_real'], all_metrics['precision_fake'] = precision_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['precision_real'], all_metrics['precision_fake'] = -1, -1
    all_metrics['acc'] = accuracy_score(y_true, y_pred)
    return all_metrics

def data2gpu(batch, use_cuda, data_type, rationale_number):
    if data_type == 'rationale':
        batch_data = {
            'content': batch[0].cuda() if use_cuda else batch[0],
            'content_masks': batch[1].cuda() if use_cuda else batch[1],
        }
        for i in range(rationale_number):
            batch_data[f'rationale_{i+1}'] = batch[2 + i].cuda() if use_cuda else batch[2 + i]
            batch_data[f'rationale_{i+1}_masks'] = batch[2 + rationale_number + i].cuda() if use_cuda else batch[2 + rationale_number + i]
        batch_data['label'] = batch[2 + 2 * rationale_number].cuda() if use_cuda else batch[2 + 2 * rationale_number]
        return batch_data
    else:
        print('error data type!')
        exit()

class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def get_monthly_path(data_type, root_path, month, data_name):
    if data_type == 'rationale':
        root_path = ensure_relative_path(root_path, 'root_path')
        return join_relative_path(root_path, data_name)
    else:
        print('No match data type!')
        exit()

def get_tensorboard_writer(config):
    if config.get('tensorboard_dir') is None:
        return None
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(dt.now())
    writer_dir = os.path.join(config['tensorboard_dir'], config['model_name'] + '_' + config['data_name'], TIMESTAMP)
    writer = SummaryWriter(logdir=writer_dir, flush_secs=5)
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    return writer

def process_test_results(test_file_path, test_res_path, label, pred, id, ae, acc):
    test_result = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    for index in range(len(label)):
        cur_res = test_data[index].copy()
        cur_res['pred'] = pred[index]
        cur_res['ae'] = ae[index]
        cur_res['acc'] = acc[index]
        test_result.append(cur_res)
    json_str = json.dumps(test_result, indent=4, ensure_ascii=False, cls=NpEncoder)
    with open(test_res_path, 'w', encoding='utf-8') as f:
        f.write(json_str)
    return