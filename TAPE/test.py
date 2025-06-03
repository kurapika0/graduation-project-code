import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
import os
import torch

data = Dataset(root='/data/fusike/TAPE/dataset', name='cora')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device='cuda').to('cuda')
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
# Setup Attack Model
model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
        attack_structure=True, attack_features=False, device='cuda', lambda_=0).to('cuda')
# Attack
model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
modified_adj = model.modified_adj

print(modified_adj)


# 2. 保存到文件
torch.save(modified_adj.cpu(), f"/data/fusike/DeepRobust/metattack/{'cora'}_{10}_structure.pt")  # 或 .pth 扩展名
print("Modified adjacency matrix saved to file.")
