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
from Models.FCNN import FCNN
from Models.MANN import Model
from Models.TCNN import TCNModel
from Models.TCNN_MOE import MANN_TCN_DynamicWeights
from Models import PAE
import torch
import torch.nn as nn
from datetime import datetime


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

def stats_calc(dataloader, PAE_model, TCNN_model, MANN_TCNN, FCNN_model, col_names,
                 k, model_str = "MANN"):
    TCNN_model.eval()
    PAE_model.eval()
    MANN_TCNN.eval()
  

    device = next(PAE_model.parameters()).device
    out_std = torch.as_tensor(dataloader.dataset.output_std, device=device, dtype=torch.float64)  # [F]

    # Running accumulators (float64 for stability)
    sum_abs   = torch.zeros_like(out_std, dtype=torch.float64)  # Σ |e|
    sum_abs2  = torch.zeros_like(out_std, dtype=torch.float64)  # Σ |e|^2  (for std of abs error)
    sum_sq    = torch.zeros_like(out_std, dtype=torch.float64)  # Σ e^2    (for RMSE)
    sum_sq_r2    = torch.zeros_like(out_std, dtype=torch.float64)  # Σ e^2    (for RMSE)
    count     = torch.tensor(0, dtype=torch.int64, device=device)  # total samples per feature
    sum_y_norm   = torch.zeros_like(out_std, dtype=torch.float64, device=device)   # Σ y_norm
    sum_y2_norm  = torch.zeros_like(out_std, dtype=torch.float64, device=device)   # Σ y_norm^2



    with torch.no_grad():
        for batch in dataloader:
            
            PAE_inputs, FCNN_inputs, FCNN_outputs = batch
            
            
            FCNN_inputs = utility.ToDevice(FCNN_inputs)
            B, F, T = FCNN_inputs.shape
            
            time_step_input = torch.full((B, 1, T), k, device=FCNN_inputs.device, dtype=torch.long)
            FCNN_inputs = torch.cat([FCNN_inputs,time_step_input], dim=1)
            FCNN_outputs = utility.ToDevice(FCNN_outputs[:,:,k]).squeeze(-1)
            
            
            if model_str == "MANN":
            # print("PAE inputs size: ", PAE_inputs.shape)
            # print("FNN inputs last slice ", FCNN_inputs[-1,:,-1])
            # print("FNN outputs:", FCNN_outputs[-1,:])
            
                PAE_inputs = utility.ToDevice(PAE_inputs)

                PAE_model.eval()
                _, _, _, params  = PAE_model(PAE_inputs)

                # # Flattening the inputs for the motion prediction network
                # flattened_inputs = utility.ToDevice(FCNN_inputs.reshape(FCNN_inputs.shape[0], -1))

                params_cat = torch.cat(params, dim=2)
                # phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
                phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
                phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])

                phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 
                phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)
                
                y_pred = MANN_TCNN(phaseInputs, FCNN_inputs)
                
            elif model_str == "TCNN":
                y_pred = TCNN_model(FCNN_inputs)
            else :
                FCNN_inputs = FCNN_inputs.reshape(FCNN_inputs.size(0), -1)
                y_pred = FCNN_model(FCNN_inputs)
                
                
                
            
            
            # TCNN 1 Step prediction
            # y_pred = model(utility.ToDevice(FCNN_inputs))
            
            # TCNN future forecastor
            # y_pred = model(utility.ToDevice(FCNN_inputs))
                    

            # accumulate GT stats in normalized space
            y_norm = FCNN_outputs.to(dtype=torch.float64)          # [B,F]
            sum_y_norm  += y_norm.sum(dim=0)
            sum_y2_norm += (y_norm * y_norm).sum(dim=0)


         
            # errors in ORIGINAL scale: (pred - gt) * std
            # broadcast std over [B, F, T]
            
            # Unnormalized
            err = (y_pred - # The code snippet you provided is not complete and does not perform any
            # specific action. It seems to be a variable name "FCNN_outputs" followed
            # by some comment characters "
            FCNN_outputs).to(dtype=torch.float64) * out_std.view(1, -1)
            
            # Normalized
            # err = (y_pred - FCNN_outputs).to(dtype=torch.float64)

            abs_err = err.abs()                       # [B, F, T]
            sq_err  = err.pow(2)                      # [B, F, T]
            
            sq_err_r2  = err.pow(2)  
            abs_err2 = abs_err.pow(2)

            # reduce over batch & time -> per-feature vectors
            sum_abs   += abs_err.sum(dim=0)
            sum_abs2  += abs_err2.sum(dim=0)
            sum_sq    += sq_err.sum(dim=0)
            count     += err.shape[0]   # B*T
            
            sum_sq_r2 += sq_err_r2.sum(dim=0)
            
    # finalize metrics
    count_f = count.to(torch.float64).clamp_min(1)
    mae  = (sum_abs / count_f)                                 # per-feature
    rmse = torch.sqrt(sum_sq / count_f)                         # per-feature
    # std of absolute error (like np.std(|e|))
    mean_abs = mae
    var_abs  = (sum_abs2 / count_f) - mean_abs.pow(2)
    var_abs  = torch.clamp(var_abs, min=0.0)
    std_abs  = torch.sqrt(var_abs)
    
    
    # SS_tot in original units: sigma^2 * (sum(y_norm^2) - sum(y_norm)^2 / N)
    sigma2   = out_std.to(torch.float64) ** 2
    SS_tot   = sigma2 * (sum_y2_norm - (sum_y_norm * sum_y_norm) / count_f)

    # SS_res already in original units: sum_sq (Σ e^2 with your std applied)
    SS_res   = sum_sq_r2
    
    eps = 1e-12

    # # print nicely
    mae_np  = mae.cpu().numpy()
    std_np  = std_abs.cpu().numpy()
    rmse_np = rmse.cpu().numpy()
    # R^2 per feature; guard division by zero (flat GT curve)
    R2 = 1.0 - (SS_res / torch.clamp(SS_tot, min=eps))

    R2_np = R2.cpu().numpy()

    print("Samples per feature (total N per feature):", int(count.item()))
    for i, joint in enumerate(col_names):
        print(f"Joint {joint} MAE = {mae_np[i]:.4f}, STD = {std_np[i]:.4f}, RMSE = {rmse_np[i]:.4f}, R^2 = {R2_np[i]:.4f}") 
        
        # ---- NEW: aggregate over the first 10 features ----
    first10 = slice(0, 10)  # indices 0..9
    
    mae_first10  = mae[first10].mean().item()
    rmse_first10 = rmse[first10].mean().item()      # optional
    std_first10  = std_abs[first10].mean().item()   # optional
    
    SS_res_first10 = SS_res[first10].sum()      # Σ over the chosen features
    SS_tot_first10 = SS_tot[first10].sum()
    
    R2_first10 = 1.0 - (SS_res_first10 / torch.clamp(SS_tot_first10, min=eps))
    R2_first10_val = float(R2_first10.item())
    
    # ---- Aggregate R^2 over ALL features ----
    SS_res_all = SS_res.sum()
    SS_tot_all = SS_tot.sum()
    
    

    R2_all = 1.0 - (SS_res_all / torch.clamp(SS_tot_all, min=eps))
    R2_all_val = float(R2_all.item())
    
    if model_str == "MANN":
        print("MANN TCNN")
    elif model_str == "TCNN":
        print("TCNN")
    else :
        print("FCNN")
    print(k)
    print("Samples per feature (total N per feature):", int(count.item()))
    print(f"MAE over first 10 features: {mae_first10:.4f}")
    # If you also want these:
    print(f"RMSE over first 10 features: {rmse_first10:.4f}")
    print(f"STD(|error|) over first 10 features: {std_first10:.4f}")
    print(f"R2 (aggregate) over first 10 features: {R2_first10_val:.4f}")
    print(f"R2 (aggregate) over ALL features: {R2_all_val:.4f}")
    
    
    last10 = slice(10, 20)  # indices 0..9
    
    mae_last10  = mae[last10].mean().item()
    rmse_last10 = rmse[last10].mean().item()      # optional
    std_last10  = std_abs[last10].mean().item()   # optional
    
    print("Samples per feature (total N per feature):", int(count.item()))
    print(f"MAE over last 10 features: {mae_last10:.4f}")
    # If you also want these:
    print(f"RMSE over last 10 features: {rmse_last10:.4f}")
    print(f"STD(|error|) over last 10 features: {std_last10:.4f}")

    
    return mae_first10, mae_last10, R2_all_val, mae_np, R2_np
    
    

def main():
    
        # Data setup
    data_path = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data/all_subjects_req_sim_data.csv"
    
    test_data_path = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data/test_subjects_req_sim_data.csv"
    
    # Locomotion mode wise
    # test_data_path = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data/Testing/Turn_step_req_sim_data.csv"
    
    PAE_model_file = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Saved Models/20250904_1754_PAE training Scherpeel Dataset - 10 Subjects 10 Phases - 40 epochs.pth"
    MANN_TCNN_model_file = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Saved Models/MANN TCNN_2.pth"
    TCNN_model_file = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Saved Models/20250911_0348_Predictor training Scherpeel Dataset - all subjects -  1 step prediction (random k prediction) - TCNModel(44,20,[64, 128, 128, 256, 64],4,0.2).pth"
    FCNN_model_file = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Saved Models/FCNN-SW.pth"
    
    df = pd.read_csv(test_data_path)
    
    print("Full df Shape: ", df.shape)
    
    cols_2_get = col_2_extract()
    df_mann = df[cols_2_get]
    
    print("Extracted df Shape: ", df_mann.shape)

    subjects, conditions, all_pairs = utility.extract_pairs(df_mann, "subject", "condition")
    groups = utility.make_group_dict(df, "subject", "condition", "time")
    
    # 4) Split by pairs
    # train_pairs,val_pairs = utility.split_pairs_train_val(all_pairs, val_frac=0.15)
    
    val_ds = GroupedSequenceDataset(
    df_mann, seq_len=201, pred_len=100, stride=1,
    time_col="time", subject_col="subject", condition_col="condition",
    include_groups=all_pairs
    )
    
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
    Mann_TCNN_model =utility.ToDevice(MANN_TCN_DynamicWeights(44,20,10,[64, 128, 128, 256, 64],50,256,4,0.2,0.2))
    Mann_TCNN_model.load_state_dict(weights_mann_tcnn)
    
    weights_tcnn = torch.load(TCNN_model_file, weights_only=True)
    TCNN_model = utility.ToDevice(TCNModel(44,20,[64, 128, 128, 256, 64],4,0.2))
    TCNN_model.load_state_dict(weights_tcnn)
    
    weights_fcnn = torch.load(FCNN_model_file, weights_only=True)
    FCNN_model = utility.ToDevice(FCNN(8844,20,5,512,0.4))
    FCNN_model.load_state_dict(weights_fcnn)

    
    K = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]
    
    # print("MANN")
    # MANN_error = stats_calc(val_loader_plot, PAE_model, TCNN_model, Mann_TCNN_model, cols_2_get[43:63], k, MANN=True)
    # print("TCNN")
    # MANN_error = stats_calc(val_loader_plot, PAE_model, TCNN_model, Mann_TCNN_model, cols_2_get[43:63], k, MANN=False)
    
    
    TCNN_mae_k = []
    MANN_TCNN_mae_k = []
    FCNN_SW_mae_k = []
    
    TCNN_mae_vel_k = []
    MANN_TCNN_mae_vel_k = []
    FCNN_SW_mae_vel_k = []
    
    MANN_r2_k = []
    TCNN_r2_k = []
    FCNN_r2_k = []
    
    MANN_joint_mae = []
    TCNN_joint_mae = []
    FCNN_joint_mae = []
    
    MANN_joint_r2 = []
    TCNN_joint_r2 = []
    FCNN_joint_r2 = []
    
    for k in K:
        # mae_first10, mae_last10, R2_all_val, mae_np, R2_np  = stats_calc(val_loader_plot, PAE_model, TCNN_model, Mann_TCNN_model, FCNN_model, cols_2_get[43:63], k, model_str="MANN")
        # MANN_TCNN_mae_k.append(mae_first10)
        # MANN_TCNN_mae_vel_k.append(mae_last10)
        # MANN_r2_k.append(R2_all_val)
        # MANN_joint_mae.append(mae_np)
        # MANN_joint_r2.append(R2_np)
        
        
        
        # mae_first10, mae_last10, R2_all_val, mae_np, R2_np = stats_calc(val_loader_plot, PAE_model, TCNN_model, Mann_TCNN_model, FCNN_model, cols_2_get[43:63], k, model_str="TCNN")
        # TCNN_mae_k.append(mae_first10)
        # TCNN_mae_vel_k.append(mae_last10)
        # TCNN_r2_k.append(R2_all_val)
        # TCNN_joint_mae.append(mae_np)
        # TCNN_joint_r2.append(R2_np)
        
        mae_first10, mae_last10, R2_all_val, mae_np, R2_np = stats_calc(val_loader_plot, PAE_model, TCNN_model, Mann_TCNN_model, FCNN_model, cols_2_get[43:63], k, model_str="FCNN")
        FCNN_SW_mae_k.append(mae_first10)
        FCNN_SW_mae_vel_k.append(mae_last10)
        FCNN_r2_k.append(R2_all_val)
        FCNN_joint_mae.append(mae_np)
        FCNN_joint_r2.append(R2_np)
        
    # frames = np.arange(len(TCNN_mae_k))  # 0..N-1 frame numbers
    
    # MANN_joint_mae_np = np.stack(MANN_joint_mae, axis=0)
    # TCNN_joint_mae_np = np.stack(TCNN_joint_mae, axis=0)    
    FCNN_joint_mae_np = np.stack(FCNN_joint_mae, axis=0)
    
    # MANN_joint_r2_np = np.stack(MANN_joint_r2, axis=0)
    # TCNN_joint_r2_np = np.stack(TCNN_joint_r2, axis=0)
    FCNN_joint_r2_np = np.stack(FCNN_joint_r2, axis=0)
    
    # print("MANN Errors")
    # print(MANN_TCNN_mae_k)
    # print(MANN_TCNN_mae_vel_k)
    # print(MANN_r2_k)
    # print(MANN_joint_mae_np)
    # print(MANN_joint_r2_np)
    
    # print("-----------------------------------------------------------")
    # print('TCNN Errors')
    # print(TCNN_mae_k)
    # print(TCNN_mae_vel_k)
    # print(TCNN_r2_k)
    # print(TCNN_joint_mae_np)
    # print(TCNN_joint_r2_np)
    
    
    print("-----------------------------------------------------------")
    print('FCNN Errors')
    print(FCNN_SW_mae_k)
    print(FCNN_SW_mae_vel_k)
    print(FCNN_r2_k)
    print(FCNN_joint_mae_np)
    print(FCNN_joint_r2_np)
    

    # plt.figure(figsize=(10, 5))
    # plt.xlim(0,100)
    # # lines with dots on every point
    # plt.plot(K, MANN_TCNN_mae_k, marker="o", markersize=2, linewidth=1.6, color="blue", label="MoE TCNN")
    # plt.plot(K, TCNN_mae_k, marker="s", markersize=2, linewidth=1.6, color="red", label="TCNN")
    # plt.plot(K, FCNN_SW_mae_k, marker="^", markersize=2, linewidth=1.6, color="green", label="FCNN-SW")

    # plt.xlabel("Prediction Time Step")
    # plt.ylabel("Mean Absolute Error")
    # # plt.grid(True, which="both", alpha=0.1)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    

if __name__ == "__main__":
    raise SystemExit(main())