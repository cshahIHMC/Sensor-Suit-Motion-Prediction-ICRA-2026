######### Author - Chinmay Shah #################


## Imports
import wandb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from Models import PAE
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import torch
from typing import List, Tuple, Dict
from DataLoader.data_loader_pae import GroupedBatchSampler, GroupedSequenceDataset


def col_2_extract():
    
    cols = ["Pelvis_V_GYROX", "Pelvis_V_GYROY", "Pelvis_V_GYROZ",
            "LThigh_V_GYROX", "LThigh_V_GYROY", "LThigh_V_GYROZ",
            "RThigh_V_GYROX", "RThigh_V_GYROY", "RThigh_V_GYROZ",
            "LShank_V_GYROX", "LShank_V_GYROY", "LShank_V_GYROZ",
            "RShank_V_GYROX", "RShank_V_GYROY", "RShank_V_GYROZ",
            "LFoot_V_GYROX", "LFoot_V_GYROY", "LFoot_V_GYROZ",
            "RFoot_V_GYROX", "RFoot_V_GYROY", "RFoot_V_GYROZ",
            "subject", "condition", "time"
            ]
    
    return cols

def extract_pairs(df: pd.DataFrame,
                  subject_col: str = "subject",
                  condition_col: str = "condition") -> Tuple[List[str], List[str], List[Tuple[str,str]]]:
    subjects   = df[subject_col].dropna().unique().tolist()
    conditions = df[condition_col].dropna().unique().tolist()
    pairs = list(
        df[[subject_col, condition_col]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    return subjects, conditions, pairs

def make_group_dict(df: pd.DataFrame,
                    subject_col: str = "subject",
                    condition_col: str = "condition",
                    time_col: str = "time") -> Dict[Tuple[str,str], pd.DataFrame]:
    groups: Dict[Tuple[str, str], pd.DataFrame] = {}
    use_time = time_col in df.columns
    for key, g in df.groupby([subject_col, condition_col], sort=False):
        g = g.sort_values(time_col) if use_time else g.sort_index()
        groups[key] = g.reset_index(drop=True)
    return groups



def main():
    # Logging False
    log_wandB = False
    
    file_name = "PAE training Scherpeel Dataset"
    project_name = "ICRA 2026"
    
    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        # project_name = config["project_name"]
        # wandb.init( project=project_name, name= config["training_tag"], config=config)
    
    # Data setup
    data_path = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data/AB01_req_sim_data.csv"
    df = pd.read_csv(data_path)
    
    cols_2_get = col_2_extract()
    df_pae = df[cols_2_get]
    
    subjects, conditions, all_pairs = extract_pairs(df_pae, "subject", "condition")
    groups = make_group_dict(df, "subject", "condition", "time")
    
    train_ds = GroupedSequenceDataset(
        df, seq_len=200, stride=1,
        time_col="time", subject_col="subject", condition_col="condition",
        include_groups=all_pairs
        )
    
    train_sampler = GroupedBatchSampler(train_ds, batch_size=32, shuffle=False, drop_last=False)
    train_loader  = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    
    all_feat0 = [] 
    for batch, meta in train_loader:
        x_bt = batch[:, 1, :]   # take feature 0 → [B, T]
        # print(x_bt.shape)
        all_feat0.append(x_bt.detach().cpu().numpy().reshape(-1))  # → [B*T]
        # print(len(all_feat0))
        # break
    
    feat0 = np.concatenate(all_feat0, axis=0)   # 1D: all batches/time concatenated
    print(len(feat0))
    
    plt.figure()
    plt.plot(feat0)
    plt.title("First feature across all batches (concatenated)")
    plt.xlabel("Concatenated time index (B×T)")
    plt.ylabel("Feature 0 value")
    plt.show()
    
    
    









if __name__ == "__main__":
    raise SystemExit(main())
    