import torch
from time import time
import numpy as np


from core.GNNs.gnn_utils import EarlyStopping
from core.data_utils.load import load_data, load_gpt_preds
from core.utils import time_logger

from core.GNNs.attack_features import attack_features

LOG_FREQ = 10


class GNNTrainer():

    def __init__(self, cfg, feature_type, data, attacked_data, num_classes):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = feature_type
        self.epochs = cfg.gnn.train.epochs
        self.attack_method = cfg.attack_method
        self.ptb_rate = cfg.ptb_rate
        self.feature_ptb_rate = cfg.feature_ptb_rate
        self.poison = cfg.poison
        self.evasion = cfg.evasion
        self.attack_feature = cfg.attack_feature
        self.attack_structure = cfg.attack_structure

        self.data = data.to(self.device)
        self.attacked_data = attacked_data.to(self.device)
        self.num_classes = num_classes

        # # ! Load data
        # data, attacked_data, num_classes = load_data(
        #     self.dataset_name, use_dgl=False, use_text=False, seed=self.seed, 
        #     attack_method=self.attack_method, ptb_rate=self.ptb_rate, device=self.device)

        self.num_nodes = data.y.shape[0]
        # self.num_classes = num_classes
        # self.data.y = data.y.squeeze()

        # ! Init gnn feature
        topk = 3 if self.dataset_name == 'pubmed' else 5
        if self.feature_type == 'ogb':
            print("Loading OGB features...")
            features = self.data.x
        elif self.feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = f"/data/fusike/TAPE/core/prt_lm/{self.dataset_name}/{self.lm_model_name}-seed{self.seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 768)))
            ).to(torch.float32)
        elif self.feature_type == 'E':
            print("Loading pretrained LM features (explanations) ...")
            LM_emb_path = f"/data/fusike/TAPE/core/prt_lm/{self.dataset_name}2/{self.lm_model_name}-seed{self.seed}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 768)))
            ).to(torch.float32)
        elif self.feature_type == 'P':
            print("Loading top-k prediction features ...")
            features = load_gpt_preds(self.dataset_name, topk)
            encoder = torch.nn.Embedding(self.num_classes+1, self.hidden_dim)
            features = encoder(features)
            features = torch.flatten(features, start_dim=1)
        else:
            print(
                f'Feature type {self.feature_type} not supported. Loading OGB features...')
            self.feature_type = 'ogb'
            features = self.data.x

        # if self.feature_type != 'P': 
        attacked_features = attack_features(features=features, data=self.data, attack_method=self.attack_method, feature_ptb_rate=self.feature_ptb_rate, device=self.device)
        # attacked_features = torch.load(f'/data/fusike/TAPE/attacked_features/{self.feature_type}_{self.feature_ptb_rate}_seed{self.seed}.pt')
        torch.save(attacked_features, f'/data/fusike/TAPE/attacked_features/{self.feature_type}_{self.feature_ptb_rate}_seed{self.seed}.pt')
        # else: attacked_features = features
        # if self.feature_type == 'P': 
            # attacked_features = attacked_features.to(torch.int64)
        self.features = features.to(self.device)
        self.attacked_features = attacked_features.to(self.device)
        # self.data = data.to(self.device)
        # self.attacked_data = attacked_data.to(self.device)

        # ! Trainer init
        use_pred = self.feature_type == 'P'

        if self.gnn_model_name == "GCN":
            from core.GNNs.GCN.model import GCN as GNN
        elif self.gnn_model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        elif self.gnn_model_name == "MLP":
            from core.GNNs.MLP.model import MLP as GNN
        else:
            print(f"Model {self.gnn_model_name} is not supported! Loading MLP ...")
            from core.GNNs.MLP.model import MLP as GNN

        self.model = GNN(in_channels=self.hidden_dim*topk if use_pred else self.features.shape[1],
                         hidden_channels=self.hidden_dim,
                         out_channels=self.num_classes,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         use_pred=use_pred).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}.pt"
        self.stopper = EarlyStopping(
            patience=cfg.gnn.train.early_stop, path=self.ckpt) if cfg.gnn.train.early_stop > 0 else None
        self.loss_func = torch.nn.CrossEntropyLoss()

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

    def _forward(self, x, edge_index):
        logits = self.model(x, edge_index)  # small-graph
        return logits

    def _train(self):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        if self.poison:
            if self.attack_structure: train_data = self.attacked_data
            else: train_data = self.attacked_data
            if self.attack_feature: train_feature = self.attacked_features
            else: train_feature = self.features
        else:
            train_data = self.data
            train_feature = self.features
        logits = self._forward(train_feature, train_data.edge_index)
        loss = self.loss_func(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        train_acc = self.evaluator(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @ torch.no_grad()
    def _evaluate(self, is_val):
        self.model.eval()
        if self.evasion and not is_val:
            if self.attack_structure: eval_data = self.attacked_data
            else: eval_data = self.data
            if self.attack_feature: eval_feature = self.attacked_features
            else: eval_feature = self.features
        else:
            eval_data = self.data
            eval_feature = self.features

        logits = self._forward(eval_feature, eval_data.edge_index)
        val_acc = self.evaluator(
            logits[self.data.val_mask], self.data.y[self.data.val_mask])
        test_acc = self.evaluator(
            logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    @time_logger
    def train(self):
        # ! Training
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate(is_val=True)
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
                print(
                    f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, logits = self._evaluate(is_val=False)
        print(
            f'[{self.gnn_model_name} + {self.feature_type}] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        return logits, res
