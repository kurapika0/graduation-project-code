import numpy as np
import torch
import random

import torch_geometric
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from deeprobust.graph.data import PrePtbDataset, Dataset, Dpr2Pyg
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import DICE, Metattack
from deeprobust.graph.targeted_attack import Nettack
import torch_geometric.utils
from .load import convert_adj_to_pyg_edges
from scipy.sparse import csr_matrix
from .align import align_data_dpr_to_data

# return cora dataset as pytorch geometric Data object together with 60/20/20 split, and list of cora IDs


def get_cora_casestudy(SEED=0, attack_method='nettack', ptb_rate=1.0, device='cpu'):
    data_X, data_Y, data_citeid, data_edges = parse_cora()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('/data/fusike/TAPE/dataset', data_name,
                        transform=T.NormalizeFeatures())  # sike
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)
    # train:val:test = 6:2:2
    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])
    
    adj = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index).tocsr()
    features = csr_matrix(data_X)
    labels = data_Y
    
    # Setup Attack Model
    if attack_method in ['DICE', 'dice']: 
        attack_method = 'dice'
        model = DICE()
        n_perturbations = int(ptb_rate * (int(data.edge_index.shape[1])))
        model.attack(ori_adj=adj, labels=data.y, n_perturbations=n_perturbations)
        modified_adj = model.modified_adj
    # elif attack_method in ['nettack', 'Nettack', 'NETTACK']:
    #     attack_method = 'nettack'
    #     # Setup Surrogate model
    #     surrogate = GCN(nfeat=data.x.shape[1], nclass=data.y.max().item()+1,
    #                 nhid=16, dropout=0, with_relu=False, with_bias=False, device=f'cuda:{device}').to(f'cuda:{device}')
    #     surrogate.fit(features=features, adj=adj, labels=labels, idx_train=data.train_id, idx_val=data.val_id, patience=30)
    #     # Setup Attack Model
    #     model = Nettack(model=surrogate, nnodes=data.num_nodes, 
    #         attack_structure=True, attack_features=False, device=f'cuda:{device}').to(f'cuda:{device}')
    #     # model.attack_features = True
    #     n_perturbations = int(ptb_rate * (int(data.edge_index.shape[1])))
    #     idx_unlabeled = np.union1d(data.val_id, data.test_id)
    #     model.attack(features=features, adj=adj, labels=labels, 
    #                     target_node=1, n_perturbations=n_perturbations
    #                     )
    #     modified_adj = model.modified_adj
    #     modified_features = model.modified_features
    elif attack_method in ['meta', 'Meta','metattack', 'Metattack', 'METATTACK']:
        attack_method = 'metattack'
        targeted = False
        data_dpr = Dataset(root='/data/fusike/DeepRobust/tmp/', name='cora', setting='prognn')
        adj, features, labels = data_dpr.adj, data_dpr.features, data_dpr.labels
        attacked_data = PrePtbDataset(root=f'/data/fusike/attacked graphs/{attack_method}', name=data_name,
                                        attack_method=attack_method,
                                        ptb_rate=ptb_rate) # here ptb_rate means number of perturbation per nodes
        data_dpr.adj = attacked_data.adj
        attacked_data = Dpr2Pyg(data_dpr)
        attacked_data = attacked_data[0]
        attacked_edge_index, targeted_nodes = align_data_dpr_to_data(data, attacked_data, targeted, data_name)

    elif attack_method in ['nettack', 'Nettack', 'NETTACK']:
        data_dpr = Dataset(root='/data/fusike/DeepRobust/tmp/', name='cora', setting='prognn')
        adj, features, labels = data_dpr.adj, data_dpr.features, data_dpr.labels
        idx_train, idx_val, idx_test = data_dpr.idx_train, data_dpr.idx_val, data_dpr.idx_test
        attacked_data = PrePtbDataset(root=f'/data/fusike/attacked graphs/{attack_method}', name=data_name,
                                        attack_method=attack_method,
                                        ptb_rate=ptb_rate) # here ptb_rate means number of perturbation per nodes
        data_dpr.adj = attacked_data.adj
        attacked_data = Dpr2Pyg(data_dpr)
        attacked_data = attacked_data._data
        attacked_edge_index, targeted_nodes = align_data_dpr_to_data(data, attacked_data, True, data_name)

        # split again
        data.test_id = np.sort(targeted_nodes)
        data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])
        # 获取非测试节点（~data.test_mask）
        non_test_nodes = torch.arange(data.num_nodes)[~data.test_mask]

        # 计算 train 和 val 的数量
        train_size = int(0.6 * data.num_nodes)
        val_size = int(0.2 * data.num_nodes)

        # 随机打乱非测试节点并划分
        perm = torch.randperm(len(non_test_nodes))
        data.train_id = non_test_nodes[perm[:train_size]]
        data.val_id = non_test_nodes[perm[train_size : train_size + val_size]]

        # 生成 train_mask 和 val_mask
        data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
        data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])


    # attacked_edge_index = convert_adj_to_pyg_edges(modified_adj)
    # np.save(f"/data/fusike/DeepRobust/{attack_method}/{data_name}_{ptb_rate}_structure.npy", attacked_edge_index)
    # print("Perturbed edges saved!")
        # attacked_edge_index = np.load(f"/data/fusike/DeepRobust/{attack_method}/{data_name}_{ptb_rate}_structure.npy", allow_pickle=True)

    attacked_data = data.clone()
    attacked_data.edge_index = torch.tensor(attacked_edge_index).long()

    return data, attacked_data, data_citeid

# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun


def parse_cora():
    path = './TAPE/dataset/cora_orig/cora'  # sike
    # path = 'dataset/cora_orig/cora'  # sike
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(use_text=False, seed=0, attack_method='nettack', ptb_rate=1.0, device='cpu'):
    data, attacked_data, data_citeid = get_cora_casestudy(seed, attack_method, ptb_rate, device)
    if not use_text:
        return data, attacked_data, None

    with open('/data/fusike/TAPE/dataset/cora_orig/mccallum/cora/papers')as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = '/data/fusike/TAPE/dataset/cora_orig/mccallum/cora/extractions/'
    text = []
    for pid in data_citeid:
        fn = pid_filename[pid]
        with open(path+fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            if 'Title:' in line:
                ti = line
            if 'Abstract:' in line:
                ab = line
        text.append(ti+'\n'+ab)
    return data, text
