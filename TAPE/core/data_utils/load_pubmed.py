import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd

from deeprobust.graph.data import PrePtbDataset, Dataset, Dpr2Pyg
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import DICE, Metattack
from deeprobust.graph.targeted_attack import Nettack
import torch_geometric.utils
from .load import convert_adj_to_pyg_edges
from scipy.sparse import csr_matrix
from .align import align_data_dpr_to_data


# return pubmed dataset as pytorch geometric Data object together with 60/20/20 split, and list of pubmed IDs


def get_pubmed_casestudy(corrected=False, SEED=0, attack_method='nettack', ptb_rate=1.0, device='cpu'):
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    data_X = normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'PubMed'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('/data/fusike/TAPE/dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    # replace dataset matrices with the PubMed-Diabetes data, for which we have the original pubmed IDs
    data.x = torch.tensor(data_X)
    data.edge_index = torch.tensor(data_edges)
    data.y = torch.tensor(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    if corrected:
        is_mistake = np.loadtxt(
            'pubmed_casestudy/pubmed_mistake.txt', dtype='bool')
        data.train_id = [i for i in data.train_id if not is_mistake[i]]
        data.val_id = [i for i in data.val_id if not is_mistake[i]]
        data.test_id = [i for i in data.test_id if not is_mistake[i]]

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
        model = DICE()
        n_perturbations = int(ptb_rate * (int(data.edge_index.shape[1])))
        model.attack(ori_adj=adj, labels=data.y, n_perturbations=n_perturbations)
        modified_adj = model.modified_adj
    elif attack_method in ['nettack', 'Nettack', 'NETTACK']:
        data_dpr = Dataset(root=f'/data/fusike/TAPE/dataset/{data_name}/raw', name='pubmed', setting='prognn')
        # data_dpr = Dataset(root=f'/data/fusike/DeepRobust/tmp/{data_name}', name='pubmed', setting='prognn')
        adj, features, labels = data_dpr.adj, data_dpr.features, data_dpr.labels
        attacked_data = PrePtbDataset(root=f'/data/fusike/attacked graphs/{attack_method}', name=data_name,
                                        attack_method=attack_method,
                                        ptb_rate=ptb_rate) # here ptb_rate means number of perturbation per nodes
        data_dpr.adj = attacked_data.adj
        attacked_data = Dpr2Pyg(data_dpr, transform=T.NormalizeFeatures())
        attacked_data = attacked_data[0]
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

    elif attack_method in ['meta', 'Meta', 'Metattack', 'METATTACK']:
        # Setup Surrogate model
        surrogate = GCN(nfeat=data.x.shape[1], nclass=data.y.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=f'cuda:{device}').to(f'cuda:{device}')
        surrogate.fit(features=features, adj=adj, labels=labels, idx_train=data.train_id, idx_val=data.val_id, patience=30)
        # Setup Attack Model
        model = Metattack(model=surrogate, nnodes=data.num_nodes, feature_shape=features.shape,
            attack_structure=True, attack_features=False, device=f'cuda:{device}', lambda_=0).to(f'cuda:{device}')
        # model.attack_features = True
        idx_unlabeled = np.union1d(data.val_id, data.test_id)
        n_perturbations = int(ptb_rate * data.edge_index.shape[1])
        model.attack(ori_features=features, ori_adj=adj, labels=labels, 
                        idx_train=data.train_id, idx_unlabeled=idx_unlabeled, n_perturbations=n_perturbations,
                        ll_constraint=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features

    # attacked_edge_index = convert_adj_to_pyg_edges(modified_adj)
    # np.save(f"/data/fusike/DeepRobust/{attack_method}/{data_name}_{ptb_rate}_structure.npy", attacked_edge_index)
    # print("perturbed edges saved!")

    attacked_data = data.clone()
    attacked_data.edge_index = torch.tensor(attacked_edge_index).long()

    return data, attacked_data, data_pubid


def parse_pubmed():
    # path = 'dataset/PubMed_orig/data/'
    path = './TAPE/dataset/PubMed_orig/data/'

    n_nodes = 19717
    n_features = 500

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}
    feature_to_index = {}

    # parse nodes
    with open(path + 'Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split('=')[-1]) - \
                1  # subtract 1 to zero-count
            data_Y[i] = label

            # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
            features = items[2:-1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

    with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')

            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
            data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
            if head != tail:
                data_edges.append(
                    (paper_to_index[head], paper_to_index[tail]))
                data_edges.append(
                    (paper_to_index[tail], paper_to_index[head]))

    return data_A, data_X, data_Y, data_pubid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_pubmed(use_text=False, seed=0, attack_method='nettack', ptb_rate=1.0, device='cpu'):
    data, attacked_data, data_pubid = get_pubmed_casestudy(SEED=seed,attack_method=attack_method, ptb_rate=ptb_rate, device=device)
    if not use_text:
        return data, attacked_data, None

    f = open('dataset/PubMed_orig/pubmed.json')
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = []
    for ti, ab in zip(TI, AB):
        t = 'Title: ' + ti + '\n'+'Abstract: ' + ab
        text.append(t)
    return data, text
