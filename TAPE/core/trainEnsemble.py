import sys
import os
import torch
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import cfg, update_cfg
from core.GNNs.ensemble_trainer import EnsembleTrainer
import pandas as pd

import time


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_acc = []
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        ensembler = EnsembleTrainer(cfg)
        acc = ensembler.train()
        all_acc.append(acc)
        del ensembler
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()
    end = time.time()

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for f in df.keys():
            df_ = pd.DataFrame([r for r in df[f]])
            print(
                f"[{f}] ValACC: {df_['val_acc'].mean():.4f} ± {df_['val_acc'].std():.4f}, TestAcc: {df_['test_acc'].mean():.4f} ± {df_['test_acc'].std():.4f}")
    print(f"Running time: {round((end-start)/len(seeds), 2)}s")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    print(cfg)
    run(cfg)
