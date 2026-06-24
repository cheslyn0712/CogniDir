import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import torch
import random
import pandas as pd
import json
import numpy as np
import nltk
import jieba
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from Src.utils.utils import ensure_relative_path

label_dict = {
    "real": 0,
    "fake": 1,
    0: 0,
    1: 1
}

def word2input(texts, max_len, tokenizer):
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def get_dataloader(path, max_len, batch_size, shuffle, bert_path, data_type,rationale_number):
    path = ensure_relative_path(path, 'path')
    bert_path = ensure_relative_path(bert_path, 'bert_path')
    local_files_only = os.path.isdir(bert_path)
    
    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=local_files_only)

    if data_type == 'rationale':
        data_list = json.load(open(path, 'r', encoding='utf-8'))
        df_data = pd.DataFrame(columns=('content', 'label') + tuple(f'rationale_{i+1}' for i in range(rationale_number)))
        
        for item in data_list:
            # Skip items with invalid labels
            if item.get('label') not in ['real', 'fake']:
                print(f"⚠️ 跳过异常样本：label={item.get('label')}")
                continue
            
            tmp_data = {}
            # content info
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            
            # rationale info
            for i in range(rationale_number):
                rationale_key = f'rationale_{i+1}'
                tmp_data[rationale_key] = item.get(rationale_key, '')

            # Convert to DataFrame and concatenate
            tmp_data = pd.DataFrame([tmp_data])
            df_data = pd.concat([df_data, tmp_data], ignore_index=True)


        content = df_data['content'].to_numpy()
        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())

        rationale_token_ids = []
        rationale_masks = []
        for i in range(rationale_number):
            rationale = df_data[f'rationale_{i+1}'].to_numpy()
            token_ids, masks = word2input(rationale, max_len, tokenizer)
            rationale_token_ids.append(token_ids)
            rationale_masks.append(masks)

        content_token_ids, content_masks = word2input(content, max_len, tokenizer)

        dataset = TensorDataset(
            content_token_ids,
            content_masks,
            *rationale_token_ids,
            *rationale_masks,
            label
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )
        return dataloader
    else:
        print('No match data type!')
        exit()