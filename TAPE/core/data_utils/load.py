import os
import json
import torch
import csv
from core.data_utils.dataset import CustomDGLDataset

import scipy.sparse as sp
import torch_geometric
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np


def convert_adj_to_pyg_edges(adj_modified):
    """根据修改后的邻接矩阵创建PyG edge_index"""
    # 确保邻接矩阵是 scipy 稀疏矩阵
    if not isinstance(adj_modified, sp.spmatrix):
        if torch.is_tensor(adj_modified):
            adj_modified = adj_modified.cpu().numpy()
        adj_modified = sp.coo_matrix(adj_modified)
    
    # 创建 PyG
    new_edge_indices = from_scipy_sparse_matrix(adj_modified)
    new_edge_indices = np.unique(new_edge_indices[0], axis=0)

    # sorted_indices = np.lexsort((new_edge_indices[1], new_edge_indices[0]))
    # sorted_edge_indices = new_edge_indices[:, sorted_indices]
    
    return new_edge_indices


def load_gpt_preds(dataset, topk):
    preds = []
    fn = f'/data/fusike/TAPE/gpt_preds/{dataset}.csv'
    print(f"Loading topk preds from {fn}")
    with open(fn, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


def load_data(dataset, use_dgl=False, use_text=False, use_gpt=False, seed=0, attack_method='nettack', ptb_rate=1.0, device='cpu'):
    if dataset == 'cora':
        from core.data_utils.load_cora import get_raw_text_cora as get_raw_text
        num_classes = 7
    elif dataset == 'pubmed':
        from core.data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
        num_classes = 3
    elif dataset == 'ogbn-arxiv':
        from core.data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
        num_classes = 40
    elif dataset == 'ogbn-products':
        from core.data_utils.load_products import get_raw_text_products as get_raw_text
        num_classes = 47
    elif dataset == 'arxiv_2023':
        from core.data_utils.load_arxiv_2023 import get_raw_text_arxiv_2023 as get_raw_text
        num_classes = 40
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        data, attacked_data, _ = get_raw_text(use_text=False, seed=seed, attack_method=attack_method, ptb_rate=ptb_rate, device=device)
        if use_dgl:
            data = CustomDGLDataset(dataset, data)
        return data, attacked_data, num_classes

    # for finetuning LM
    if use_gpt:
        data, attacked_data, text = get_raw_text(use_text=False, seed=seed)
        folder_path = 'gpt_responses/{}'.format(dataset)
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]
        text = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                text.append(content)
    # if use_gpt:
    #     data, text = get_raw_text(use_text=False, seed=seed)
    #     folder_path = 'llama_responses/{}'.format(dataset)
    #     print(f"using gpt: {folder_path}")
    #     n = data.y.shape[0]
    #     text = []
    #     for i in range(n):
    #         filename = str(i) + '.json'
    #         file_path = os.path.join(folder_path, filename)
    #         with open(file_path, 'r') as file:
    #             json_data = json.load(file)
    #             content = json_data['generation']['content']
    #             text.append(content)
    else:
        data, text = get_raw_text(use_text=True, seed=seed)

    return data, attacked_data, num_classes, text
