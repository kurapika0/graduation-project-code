import torch
import scipy.sparse as sp
import numpy as np
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.defense import GCN
from torch_geometric.utils import to_dgl, to_scipy_sparse_matrix


def attack_features(features, data, attack_method, feature_ptb_rate, device):
    # n_perturbations = int(feature_ptb_rate*features.shape[0]*features.shape[1])
    n_perturbations = int(feature_ptb_rate)
    attack_method = attack_method.lower()
   
    adj = to_scipy_sparse_matrix(data.edge_index).tocsr()
    labels = data.y.clone().cpu()
    idx_train, idx_val, idx_test = data.train_id, data.val_id, data.test_id
    features = features.detach()
    # features_ = features.to(torch.float)
    # idx_train, idx_val = idx_train.cpu().numpy(), idx_val.cpu().numpy()
    idx_unlabeled = np.concatenate((idx_val, idx_test))
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)

    feature_shape = features.shape
    # features = features.clone().numpy()
    # features = sp.lil_matrix(features)
    if attack_method in ['nettack']:
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=False,
                            attack_features=True, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, idx_test, n_perturbations, verbose=False)
    if attack_method in ['meta', 'metattack']:
        model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=feature_shape, attack_structure=False, 
                        attack_features=True, device=device)
        model = model.to(device)
        model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=False)
    
    modified_features = model.modified_features
    # data.features = modified_features

    return modified_features