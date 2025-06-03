import torch
import time
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.global_attack import DICE


def process_data(data, attack_method, ptb_rate, data_name):
    targeted_nodes = None
    if attack_method in ['DICE', 'dice']: 
        attack_method = 'dice'
        model = DICE()
        n_perturbations = int(ptb_rate * (int(data.adj.size)))
        model.attack(ori_adj=data.adj, labels=data.labels, n_perturbations=n_perturbations)
        attacked_adj = model.modified_adj
    elif attack_method in ['nettack', 'Nettack', 'NETTACK', 'meta', 'metattack', 'mettack', 'METATTACK']:
        if attack_method in ['nettack', 'Nettack', 'NETTACK']:
            attack_method = 'nettack'
            targeted = True
        else: 
            attack_method = 'metattack'
            targeted = False
        attacked_data = PrePtbDataset(root=f'/data/fusike/attacked graphs/{attack_method}', name=data_name,
                                        attack_method=attack_method,
                                        ptb_rate=ptb_rate) # here ptb_rate means number of perturbation per nodes
        attacked_adj = attacked_data.adj
        if targeted: targeted_nodes = attacked_data.target_nodes

    return attacked_adj, targeted_nodes


def train(data_name, attack_method, ptb_rate, poison, evasion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Dataset(root='/data/fusike/DeepRobust/tmp/', name=data_name, setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    attacked_adj, targeted_nodes = process_data(data=data, data_name=data_name, 
                                                attack_method=attack_method, ptb_rate=ptb_rate, )
    if attack_method in ['nettack', 'Nettack', 'NETTACK']:
        idx_test = targeted_nodes

    gcn = GCN(nfeat=features.shape[1],
        nhid=16,
        nclass=labels.max().item() + 1,
        dropout=0.5, device=device)
    gcn.initialize()
    gcn = gcn.to(device)


    # training
    if poison:
        gcn.fit(features, attacked_adj, labels, idx_train, idx_val) # train without earlystopping
    else:
        gcn.fit(features, adj, labels, idx_train, idx_val)

    if evasion:
        gcn.eval()
        output = gcn.predict(features, attacked_adj)
        # output = self.output
        output = output.cpu()
        labels = torch.LongTensor(labels, device='cpu')
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()
    else:   
        acc = gcn.test(idx_test)
    return acc
    # gcn.fit(features, adj, labels, idx_train, idx_val, patience=30) # train with earlystopping


if __name__ == '__main__':
    data_name = 'citeseer'
    poison = False
    evasion = False
    epochs = 10
    # attack_list = {'dice':[0.1, 0.3], 'nettack':[1.0, 5.0], 'metattack':[0.1, 0.25]}
    # attack_list = {'dice':[0.1, 0.3], 'nettack':[1.0, 5.0]}
    attack_list = {'nettack':[1.0]}
    start_time = time.time()
    for attack_method, ptb_rates in attack_list.items():
        for ptb_rate in ptb_rates:
            print(f'================== {attack_method} {ptb_rates} ==================')
            all_acc = []
            for i in tqdm(range(epochs)):
                acc = train(data_name, attack_method, ptb_rate, poison, evasion)
                all_acc.append(acc)
            print(f"Final Acc: {np.mean(all_acc):.4f}±{np.std(all_acc):.4f}")

            with open("/data/fusike/DeepRobust/output.txt", 'a') as fp:
                fp.write(f'Attack Method: {attack_method} Ptb Rate: {ptb_rate} Poison: {poison}  Final Acc: {np.mean(all_acc):.4f}±{np.std(all_acc):.4f}\n')
    end_time = time.time()
    running_time = end_time - start_time
    with open("/data/fusike/DeepRobust/output.txt", 'a') as fp:
        fp.write(f"Time: {running_time}s\n")

    