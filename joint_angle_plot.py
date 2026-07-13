"""
Loads a trained PAE + MANN checkpoint pair and plots predicted vs. ground-truth
joint angle trajectories over a fixed time window, for paper figures.

Author: Chinmay Shah
Institution: Institute for Human and Machine Cognition (IHMC) / University of West Florida (UWF)
"""

from Library import utility
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from DataLoader.data_loader_pae import GroupedBatchSampler, GroupedSequenceDataset
from torch.utils.data import Dataset, DataLoader, Subset
from Models.FCNN import FCNN
from Models.MANN import Model
from Models.TCNN import TCNModel
from Models.TCNN_MOE import MANN_TCN_DynamicWeights
from Models import PAE
import torch
import torch.nn as nn
import os
from datetime import datetime

# Repo root (folder this file lives in) - used so the default paths below work
# regardless of where the repository is cloned.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def col_2_extract():
    """Return the ordered list of IMU/kinematic/velocity feature and label columns used by this script."""
    
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

def plot_results(dataloader, PAE_model, model, col_names, train_loader):
    """Run the PAE + MANN model pair over `dataloader` and plot unnormalized hip-angle
    predictions vs. ground truth for a fixed time window (used to generate paper figures)."""
    model.eval()
    PAE_model.eval()
    
    # save_csv_path = "<repo>/Data/params_normalization.csv"
    
    k = 0
    ground_truth = []
    preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            
            PAE_inputs, FCNN_inputs, FCNN_outputs = batch
            # print("PAE inputs size: ", PAE_inputs.shape)
            # print("FNN inputs last slice ", FCNN_inputs[-1,:,-1])
            # print("FNN outputs:", FCNN_outputs[-1,:])
            
            PAE_inputs = utility.ToDevice(PAE_inputs)
            
            
            _, _, _, params  = PAE_model(PAE_inputs)
            
            params_cat = torch.cat(params, dim=2)
            # phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
            phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
            phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])
            
            phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 
 
            phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)
            
            
            FCNN_inputs = utility.ToDevice(FCNN_inputs)
            # B, F, T = FCNN_inputs.shape
            # time_step_input = torch.full((B, 1, T), k, device=FCNN_inputs.device, dtype=torch.long)
            # FCNN_inputs = torch.cat([FCNN_inputs,time_step_input], dim=1)
            FCNN_outputs = utility.ToDevice(FCNN_outputs[:,:,k]).squeeze(-1)

             # MANN
            y_pred = model(phaseInputs, FCNN_inputs)
            

            output_np = utility.Item(FCNN_outputs).squeeze(-1).numpy()
            pred_np = utility.Item(y_pred).numpy()

            preds.append(pred_np)
            ground_truth.append(output_np)
                  

    # Concatenate all batch outputs
    pred_all = np.concatenate(preds, axis=0)         # shape: (N, 1, 28)
    ground_truth_all = np.concatenate(ground_truth, axis=0)  # shape: (N, 1, 28)
    
    # print(pred_all.shape)
    # print(ground_truth_all.shape)
    
    # Unnormalize the values
    pred_all_unnormalized = pred_all * train_loader.dataset.output_std.to_numpy() + train_loader.dataset.output_mean.to_numpy()
    ground_truth_all_unnormalized = ground_truth_all * train_loader.dataset.output_std.to_numpy() + train_loader.dataset.output_mean.to_numpy()
    
    # pred_all_unnormalized = pred_all
    # ground_truth_all_unnormalized = ground_truth_all
    
    joints = col_names     
    # Assume pred and ground_truth are (N, 20)
    N, num_features = pred_all_unnormalized.shape

    rows = num_features // 2   # 10 rows if 20 features
    cols = 2
    
    
    t1 = 1616
    t2 = 1975
    
    # Hip Angle
    plt.figure(figsize=(10, 7.5))

    # lines with dots on every point
    plt.plot(ground_truth_all_unnormalized[t1:t2, 0], linewidth=2, color="black", alpha = 0.5, label="Ground Truth")
    plt.plot(pred_all_unnormalized[t1:t2, 0], linewidth=3, color="black", label="Prediction")


    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)    
    plt.grid(True, which="both", alpha=0.1)
    plt.legend()
    plt.tight_layout()
    plt.show()


    
    
    ## Knee Angle
    plt.figure(figsize=(5, 2.5))

    # lines with dots on every point
    plt.plot(ground_truth_all_unnormalized[t1:t2, 3], linewidth=2, color="black", alpha = 0.5, label="Ground Truth")
    plt.plot(pred_all_unnormalized[t1:t2, 3], linewidth=3, color="black", label="Prediction")


    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)    
    plt.grid(True, which="both", alpha=0.1)
    plt.tight_layout()
    plt.show()
    
    
    ## Ankle Angle
    plt.figure(figsize=(5, 2.5))

    # lines with dots on every point
    plt.plot(ground_truth_all_unnormalized[t1:t2, 4], linewidth=2, color="black", alpha = 0.5, label="Ground Truth")
    plt.plot(pred_all_unnormalized[t1:t2, 4], linewidth=3, color="black", label="Prediction")


    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)    
    plt.grid(True, which="both", alpha=0.1)
    plt.tight_layout()
    plt.show()
    
    
    ## Hip Velocity
    plt.figure(figsize=(5, 2.5))

    # lines with dots on every point
    plt.plot(ground_truth_all_unnormalized[t1:t2, 10], linewidth=2, color="black", alpha = 0.5, label="Ground Truth")
    plt.plot(pred_all_unnormalized[t1:t2, 10], linewidth=3, color="black", label="Prediction")


    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)    
    plt.grid(True, which="both", alpha=0.1)
    plt.tight_layout()
    plt.show()
    
    ## Knee Velocity
    plt.figure(figsize=(5, 2.5))

    # lines with dots on every point
    plt.plot(ground_truth_all_unnormalized[t1:t2, 13], linewidth=2, color="black", alpha = 0.5, label="Ground Truth")
    plt.plot(pred_all_unnormalized[t1:t2, 13], linewidth=3, color="black", label="Prediction")


    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)    
    plt.grid(True, which="both", alpha=0.1)
    plt.tight_layout()
    plt.show()
    
    ## Ankle Velocity
    plt.figure(figsize=(5, 2.5))

    # lines with dots on every point
    plt.plot(ground_truth_all_unnormalized[t1:t2, 14], linewidth=2, color="black", alpha = 0.5, label="Ground Truth")
    plt.plot(pred_all_unnormalized[t1:t2, 14], linewidth=3, color="black", label="Prediction")


    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)    
    plt.grid(True, which="both", alpha=0.1)
    plt.tight_layout()
    plt.show()
    
    
    # Number of features you want in each figure
    # features_per_plot = 10  

    # # Loop over feature groups
    # for group in range((num_features + features_per_plot - 1) // features_per_plot):
    #     start = group * features_per_plot
    #     end = min((group + 1) * features_per_plot, num_features)

    #     # Create a grid: here 2 rows x 5 cols for 10 features
    #     rows, cols = 2, 5
    #     fig, axes = plt.subplots(rows, cols, figsize=(15, 8), sharex=True, sharey=True)
    #     axes = axes.flatten()

    #     for i, feat_idx in enumerate(range(start, end)):
    #         ax = axes[i]
    #         ax.plot(pred_all_unnormalized[:, feat_idx], label="Prediction", linewidth=1.2, color="red")
    #         ax.plot(ground_truth_all_unnormalized[:, feat_idx], label="Ground Truth", linewidth=1.0, color="black", alpha=0.7)
    #         # Use joint name instead of generic index
    #         ax.set_title(joints[feat_idx])

    #         if feat_idx == 0:  # only first subplot gets the legend
    #             ax.legend(loc="upper right")

    #     # Hide any unused subplots in the last group
    #     for j in range(i+1, len(axes)):
    #         fig.delaxes(axes[j])

    #     plt.xlabel("Time (samples)")
    #     plt.tight_layout()
    #     plt.show()
    

def main():
    
        # Data setup
    data_path = os.path.join(BASE_DIR, "Data", "all_subjects_req_sim_data.csv")

    test_data_path = os.path.join(BASE_DIR, "Data", "Testing", "Step_ups_req_sim_data.csv")

    # Locomotion mode wise
    # test_data_path = "<repo>/Data/Testing/Turn_step_req_sim_data.csv"

    PAE_model_file = os.path.join(BASE_DIR, "Saved Models", "20250904_1754_PAE training Scherpeel Dataset - 10 Subjects 10 Phases - 40 epochs.pth")
    MANN_TCNN_model_file = os.path.join(BASE_DIR, "Saved Models", "MANN 1 Step.pth")
    TCNN_model_file = os.path.join(BASE_DIR, "Saved Models", "20250911_0348_Predictor training Scherpeel Dataset - all subjects -  1 step prediction (random k prediction) - TCNModel(44,20,[64, 128, 128, 256, 64],4,0.2).pth")
    FCNN_model_file = os.path.join(BASE_DIR, "Saved Models", "FCNN-SW.pth")
    
    
    df_train = pd.read_csv(data_path)
    df_val = pd.read_csv(test_data_path)
    
    print("Full df Shape: ", df_train.shape)
    
    cols_2_get = col_2_extract()
    df_mann_train = df_train[cols_2_get]
    df_mann_val = df_val[cols_2_get]
    
    print("Extracted df Shape: ", df_mann_val.shape)

    subjects, conditions, val_all_pairs = utility.extract_pairs(df_mann_val, "subject", "condition")
    subjects, conditions, train_all_pairs = utility.extract_pairs(df_mann_train, "subject", "condition")
    
    # 4) Split by pairs
    # train_pairs,val_pairs = utility.split_pairs_train_val(all_pairs, val_frac=0.15)
    
    # 4) Split by pairs
    # train_pairs,val_pairs = utility.split_pairs_train_val(all_pairs, val_frac=0.15)
    
    train_ds = GroupedSequenceDataset(
        df_mann_train, seq_len=201, pred_len=100, stride=1,
        time_col="time", subject_col="subject", condition_col="condition",
        include_groups=train_all_pairs
        )
        
    val_ds = GroupedSequenceDataset(
    df_mann_val, seq_len=201, pred_len=100, stride=1,
    time_col="time", subject_col="subject", condition_col="condition",
    include_groups=val_all_pairs
    )
    
    train_sampler = GroupedBatchSampler(train_ds, batch_size=64, shuffle=False, drop_last=False)
    train_loader  = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    
    val_sampler_plot = GroupedBatchSampler(val_ds, batch_size=64, shuffle=False, drop_last=False)
    val_loader_plot  = DataLoader(val_ds, batch_sampler=val_sampler_plot, num_workers=8, pin_memory=True)
    
    ## Load PAE file
    weights = torch.load(PAE_model_file, weights_only=True)
    PAE_model = utility.ToDevice(PAE.Model(
                          input_channels=21,
                          embedding_channels=10,
                          intermediate_channels=16,
                          time_range=201,
                          window=1.0
                         ))
    
    PAE_model.load_state_dict(weights)
    
    weights_mann_tcnn = torch.load(MANN_TCNN_model_file, weights_only=True)
    Mann_TCNN_model =utility.ToDevice(MANN_TCN_DynamicWeights(43,20,10,[64, 128, 128, 256, 64],50,256,4,0.2,0.2))
    Mann_TCNN_model.load_state_dict(weights_mann_tcnn)
    
    plot_results(val_loader_plot, PAE_model=PAE_model, model=Mann_TCNN_model, col_names=cols_2_get[43:63], train_loader=train_loader)
    


if __name__ == "__main__":
    raise SystemExit(main())