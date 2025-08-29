############# Author -Chinmay Shah ##################

# Train Predictor
## Imports
import wandb
from Library import utility
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from DataLoader.data_loader_pae import GroupedBatchSampler, GroupedSequenceDataset
from torch.utils.data import Dataset, DataLoader, Subset


def col_2_extract():
    
    cols = ["Pelvis_V_GYROX", "Pelvis_V_GYROY", "Pelvis_V_GYROZ",
            "LThigh_V_GYROX", "LThigh_V_GYROY", "LThigh_V_GYROZ",
            "RThigh_V_GYROX", "RThigh_V_GYROY", "RThigh_V_GYROZ",
            "LShank_V_GYROX", "LShank_V_GYROY", "LShank_V_GYROZ",
            "RShank_V_GYROX", "RShank_V_GYROY", "RShank_V_GYROZ",
            "LFoot_V_GYROX", "LFoot_V_GYROY", "LFoot_V_GYROZ",
            "RFoot_V_GYROX", "RFoot_V_GYROY", "RFoot_V_GYROZ",
            
            "Pelvis_V_ACCX", "Pelvis_V_ACCY", "Pelvis_V_ACCZ",
            "LThigh_V_ACCX", "LThigh_V_ACCY", "LThigh_V_ACCZ",
            "RThigh_V_ACCX", "RThigh_V_ACCY", "RThigh_V_ACCZ",
            "LShank_V_ACCX", "LShank_V_ACCY", "LShank_V_ACCZ",
            "RShank_V_ACCX", "RShank_V_ACCY", "RShank_V_ACCZ",
            "LFoot_V_ACCX", "LFoot_V_ACCY", "LFoot_V_ACCZ",
            "RFoot_V_ACCX", "RFoot_V_ACCY", "RFoot_V_ACCZ",
            "weight",

            "hip_flexion_r" ,"hip_adduction_r" ,"hip_rotation_r",
            "knee_angle_r", "ankle_angle_r" ,
            "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
            "knee_angle_l", "ankle_angle_l",
            "hip_flexion_velocity_r", "hip_adduction_velocity_r", "hip_rotation_velocity_r",
            "knee_velocity_r", "ankle_velocity_r",
            "hip_flexion_velocity_l", "hip_adduction_velocity_l", "hip_rotation_velocity_l",
            "knee_velocity_l", "ankle_velocity_l",

            "subject", "condition"
            ]
    
    return cols

def main():
    
    # Different Flags
    # Logging False
    log_wandB = False
    train_and_plot = False
    
    file_name = "Predictor training Scherpeel Dataset"
    project_name = "ICRA 2026"
    
    # Config the configurations
    config = {
        "training_tag": file_name,
        "project_name": project_name,
        "epochs": 1,
        "batch_size": 32,
        "num_workers": 8,
        "momentum":0.9,
        "lr": 1e-4,
        "dropout": 0.0,
        "dataset": "IHMC Senorsuit",
        "seq_length": 201,
        "inputs": 21,
        "outputs": 21
    }

    
    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        project_name = config["project_name"]
        wandb.init( project=project_name, name= config["training_tag"], config=config)
    
    # Data setup
    data_path = "/home/chinmay/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data/AB01_req_sim_data.csv"
    df = pd.read_csv(data_path)
    
    print("Full df Shape: ", df.shape)
    
    cols_2_get = col_2_extract()
    df_mann = df[cols_2_get]
    
    print("Extracted df Shape: ", df_mann.shape)

    subjects, conditions, all_pairs = utility.extract_pairs(df_mann, "subject", "condition")
    groups = utility.make_group_dict(df, "subject", "condition", "time")
    
    # 4) Split by pairs
    # train_pairs,val_pairs = utility.split_pairs_train_val(all_pairs, val_frac=0.15)
    
    train_ds = GroupedSequenceDataset(
        df_mann, seq_len=config["seq_length"], pred_len=1, stride=1,
        time_col="time", subject_col="subject", condition_col="condition",
        include_groups=all_pairs
        )
    
    train_sampler = GroupedBatchSampler(train_ds, batch_size=1, shuffle=False, drop_last=False)
    train_loader  = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=8, pin_memory=True)


    all_feat0 = [] 
    i = 0
    for batch in train_loader:

        PAE_input, MANN_input, MANN_output = batch
        print("PAE_input Shape: ", PAE_input.shape)
        print("MANN_input Shape: ", MANN_input.shape)
        print("MANN_output Shape: ", MANN_output.shape)
        break
        # x_bt = batch[:, 1, :]   # take feature 0 → [B, T]
        # # print(x_bt.shape)
        # all_feat0.append(x_bt.detach().cpu().numpy().reshape(-1))  # → [B*T]
        # # print(len(all_feat0))
        # # break
        
        # i = i+1
        
        # if i%1000 == 0:
        #     print(i)
        # if i == 20000:
        #     break
    
    # feat0 = np.concatenate(all_feat0, axis=0)   # 1D: all batches/time concatenated
    # print(len(feat0))
    
    # plt.figure()
    # plt.plot(feat0)
    # plt.title("First feature across all batches (concatenated)")
    # plt.xlabel("Concatenated time index (B×T)")
    # plt.ylabel("Feature 0 value")
    # plt.show()




if __name__ == "__main__":
    raise SystemExit(main())