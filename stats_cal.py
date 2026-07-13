"""
Loads a trained checkpoint (TCNN, FCNN_SW, MANN, MoETCNN, or LSTM) and a held-out
test-subject CSV, then evaluates per-joint prediction statistics at a given
prediction horizon. Used to generate the final test-set results reported in the
paper.

Update `model_file_path` (and the paths inside `run_list`) to point at your local
checkpoint/data locations before running (see README).

Author: Chinmay Shah
Institution: Institute for Human and Machine Cognition (IHMC) / University of West Florida (UWF)
"""

from Library import utility
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from DataLoader.data_loader_pae import GroupedBatchSampler, GroupedSequenceDataset
from torch.utils.data import Dataset, DataLoader, Subset
from Models import PAE
from Models.MANN import Model
from Models.TCNN_MOE import MANN_TCN_DynamicWeights, MANN_TCN_DynamicWeights_Forecast
from Models.FCNN import FCNN
from Models.TCNN import TCNModel, TCNModel_Forecast
from Models.LSTM import LSTM
import matplotlib.pyplot as plt

# Repo root (folder this file lives in) - used so the default paths below work
# regardless of where the repository is cloned.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_file_path = os.path.join(BASE_DIR, "Saved Models", "20260302_1550_MANN for final Paper test Abalation MANN without expert weights k = 100.pth")
# pae_model_file_path = "<repo>/Saved Models/20260212_0245_PAE on SS Dataset - 10 Subjects - 200 epochs (256 batch size).pth"
# data_file_path = "<repo>/Data - Second Skin/Testing/test_subject_req_data.csv"
# prediction_horizon = 100



# model_name = "MANN"


# List of models for k evaluation
# model_list = {
#     1 : "<repo>/Saved Models/20260224_1938_LSTM for final Paper test k = 1.pth",
#     5 : "<repo>/Saved Models/20260225_0052_LSTM for final Paper test k = 5.pth",
#     20: "<repo>/Saved Models/20260225_0607_LSTM for final Paper test k = 20.pth",
#     50: "<repo>/Saved Models/20260225_1321_LSTM for final Paper test k = 50.pth",
#     80: "<repo>/Saved Models/20260225_1836_LSTM for final Paper test k = 80.pth",
#     100: "<repo>/Saved Models/20260225_2350_LSTM for final Paper test k = 100.pth"
# }


def run_list(model_file_path, prediction_horizon, model_name):
    """Load `model_name`'s checkpoint at `model_file_path`, run it on the held-out test
    subject data at `prediction_horizon`, and print/plot the resulting statistics."""

    pae_model_file_path = os.path.join(BASE_DIR, "Saved Models", "20260212_0245_PAE on SS Dataset - 10 Subjects - 200 epochs (256 batch size).pth")
    # TODO: add the path to your held-out test-subject CSV here (produced by
    # running data_extraction.py on your local copy of the dataset - see README).
    data_file_path = os.path.join(BASE_DIR, "Data - Second Skin", "Testing", "<your_test_dataset>.csv")

    future_forcast = False
    
    if prediction_horizon != 1:
        future_forcast = True


    # Load the data
    
    df = pd.read_csv(data_file_path)
    print("Dataset Size: ", df.shape)
    
    df[df.columns[56:66]].plot()
    plt.show()
    subjects, conditions, all_pairs = utility.extract_pairs(df, "subject", "condition")

    # print("All Pairs:", all_pairs)
    ds = GroupedSequenceDataset(
            df, seq_len=201, pred_len=prediction_horizon, stride=1,
            subject_col="subject", condition_col="condition",
            include_groups=all_pairs
            )

    dataloader = DataLoader(ds, shuffle=False, num_workers=16, batch_size=32, pin_memory=True)

    weights = torch.load(model_file_path, weights_only=True)

    if model_name =="TCNN":

        if future_forcast:
            model = utility.ToDevice(TCNModel_Forecast(
                                            input_size=46,
                                            output_size=20,
                                            horizon=prediction_horizon,
                                            num_channels=[80, 80, 80, 80, 80],
                                            kernel_size=6,
                                            dropout=0.2))
        else:   
            model = utility.ToDevice(TCNModel(
                                            input_size=46,
                                            output_size=20,
                                            num_channels=[80, 80, 80, 80, 80],
                                            kernel_size=6,
                                            dropout=0.2))

    elif model_name == "FCNN_SW":

        model = utility.ToDevice(FCNN(
                                inputs=46, 
                                outputs=20, 
                                numOfLayers=3, 
                                hiddenDimension=256, 
                                input_seq_len=201, 
                                predictionHorizon=prediction_horizon, 
                                dropoutRate=0.2))
    elif model_name == "MANN":

        ## Load PAE file
        pae_weights = torch.load(pae_model_file_path, weights_only=True)
        PAE_model = utility.ToDevice(PAE.Model(
                          input_channels=21,
                          embedding_channels=10,
                          intermediate_channels=16,
                          time_range=201,
                          window=1.0
                         ))
        PAE_model.load_state_dict(pae_weights)

        # Setting up a mode adaptive neural network MANN
        model = utility.ToDevice(Model( gating_input=50,
                                               gating_hidden=256,
                                               gating_output=10,
                                               main_input=46*50,
                                               main_hidden=256,
                                               main_output=20,
                                               prediction_horizon=prediction_horizon,
                                               dropout=0.2))

    elif model_name == "LSTM":

        model = utility.ToDevice(LSTM(input_size=46,
                                              hidden_size=256,
                                              num_layers=3,
                                              output_size=20,
                                              pred_horizon=prediction_horizon,
                                              dropout=0.2))


    model.load_state_dict(weights)

    # Generate the Stats
    model.eval()

    out_std = torch.as_tensor(dataloader.dataset.output_std, dtype=torch.float64)  # [F]

    # Running accumulators (float64 for stability)
    sum_abs   = torch.zeros_like(out_std, dtype=torch.float64)  # Σ |e|
    sum_abs2  = torch.zeros_like(out_std, dtype=torch.float64)  # Σ |e|^2  (for std of abs error)
    sum_sq    = torch.zeros_like(out_std, dtype=torch.float64)  # Σ e^2    (for RMSE)
    count     = torch.tensor(0, dtype=torch.int64)  # total samples per feature
    sum_y     = torch.zeros_like(out_std, dtype=torch.float64)  # Σ y
    sum_y2    = torch.zeros_like(out_std, dtype=torch.float64)  # Σ y^2

    ground_truth = []
    preds = []



    with torch.no_grad():
        for batch in dataloader:

            Autoencoder_inputs, Predictor_inputs, Predictor_outputs = batch

            Autoencoder_inputs = utility.ToDevice(Autoencoder_inputs)
            Predictor_input_gpu = utility.ToDevice(Predictor_inputs)

            # 1-Time Prediction
            Predictor_outputs = Predictor_outputs.squeeze(-1)  # Remove the pred_length dimension if it's 1, shape becomes [batch_size, output_features]


            if model_name == "MANN":

                # PAE_model.eval()
                # params  = PAE_model(Autoencoder_inputs)

                # params_cat = torch.cat(params, dim=2)
                # phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
                # phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
                # phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])

                # phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 

                # phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)

                # Flattening the inputs for the motion prediction network
                # flattened_inputs = utility.ToDevice(Predictor_input.reshape(Predictor_input.shape[0], -1))

                # Only using the last 20 time steps to predict the future time step
                last_step_inputs = Predictor_input_gpu[:, :, -50:]              # shape = [batch, features]
                flattened_inputs = utility.ToDevice(last_step_inputs.reshape(last_step_inputs.shape[0], -1)) # already flat
                phaseInputs = torch.ones(Predictor_input_gpu.shape[0], 50).to(flattened_inputs.device)  # Dummy phase inputs of shape (B, 50)

                y_pred, _ = model(phaseInputs, flattened_inputs)
            elif model_name == "FCNN_SW":
                Predictor_input_gpu_flat = Predictor_input_gpu.reshape(Predictor_input_gpu.shape[0], -1)
                # FCNN sliding window prediction
                y_pred = model(Predictor_input_gpu_flat)    

            elif model_name == "TCNN":
                # TCNN 1 Step prediction
                y_pred = model(Predictor_input_gpu)

                # errors in ORIGINAL scale: (pred - gt) * std
                # broadcast std over [B, F, T]
            elif model_name == "LSTM":
                y_pred = model(Predictor_input_gpu)



            if future_forcast:
            
            
                # Ground truth in original scale
                y_true = Predictor_outputs.to(dtype=torch.float64) * out_std.view(1, -1, 1)
                reduce_dims = (0, 2)

                # print("y: ", y_true.shape[0])

                # accumulate
                sum_y  += y_true.sum(dim=reduce_dims)
                sum_y2 += (y_true.pow(2)).sum(dim=reduce_dims)

                err = (utility.Item(y_pred) - Predictor_outputs).to(dtype=torch.float64) * out_std.view(1, -1, 1)

                abs_err = err.abs()                       # [B, F, T]
                sq_err  = err.pow(2)                      # [B, F, T]
                abs_err2 = abs_err.pow(2)

                # reduce over batch & time -> per-feature vectors

                sum_abs   += abs_err.sum(dim=reduce_dims)
                sum_abs2  += abs_err2.sum(dim=reduce_dims)
                sum_sq    += sq_err.sum(dim=reduce_dims)
                count     += err.shape[0] * err.shape[2]  # B*T

            else:
                output_np = utility.Item(Predictor_outputs).numpy()
                pred_np = utility.Item(y_pred).numpy()

                preds.append(pred_np)
                ground_truth.append(output_np)

        if future_forcast:
            # finalize metrics
            count_f = count.to(torch.float64).clamp_min(1)
            mae  = (sum_abs / count_f)                                 # per-feature
            rmse = torch.sqrt(sum_sq / count_f)                         # per-feature
            # std of absolute error (like np.std(|e|))
            mean_abs = mae
            var_abs  = (sum_abs2 / count_f) - mean_abs.pow(2)
            var_abs  = torch.clamp(var_abs, min=0.0)
            std_abs  = torch.sqrt(var_abs)

            # total variance per feature
            mean_y = sum_y / count_f
            ss_tot = sum_y2 - count_f * mean_y.pow(2)

            # avoid divide-by-zero
            ss_tot = torch.clamp(ss_tot, min=1e-12)

            r2 = 1.0 - (sum_sq / ss_tot)
            r2_np = r2.cpu().numpy()


            # print nicely
            mae_np  = mae.cpu().numpy()
            std_np  = std_abs.cpu().numpy()
            rmse_np = rmse.cpu().numpy()

            # print("Samples per feature (total N per feature):", int(count.item()))
            # print("----- Joint Angle Metrics -----")
            # for i, joint in enumerate(df.columns[46:66]):
            #     if i == 10:
            #         print("----- Joint Moment Metrics -----")
            #     print(f"Joint {joint} MAE = {mae_np[i]:.4f}, STD = {std_np[i]:.4f}, RMSE = {rmse_np[i]:.4f}, R2 = {r2_np[i]:.4f}")    

            # ---- Split indices ----
            angle_idx = slice(0, 10)
            moment_idx = slice(10, 20)

            # ---- Aggregate metrics ----
            angle_mae  = mae_np[angle_idx].mean()
            angle_std  = std_np[angle_idx].mean()
            angle_rmse = rmse_np[angle_idx].mean()
            angle_r2   = 1.0 - (sum_sq[:10].sum() / ss_tot[:10].sum())

            moment_mae  = mae_np[moment_idx].mean()
            moment_std  = std_np[moment_idx].mean()
            moment_rmse = rmse_np[moment_idx].mean()
            moment_r2   = 1.0 - (sum_sq[10:20].sum() / ss_tot[10:20].sum())

            print("Samples per feature (total N per feature):", int(count.item()))
            print("========================================")
            print("Joint Angle Metrics (First 10 Features)")
            print(f"MAE = {angle_mae:.4f}, STD = {angle_std:.4f}, RMSE = {angle_rmse:.4f}, R2 = {angle_r2:.4f}")

            print("========================================")
            print("Joint Moment Metrics (Next 10 Features)")
            print(f"MAE = {moment_mae:.4f}, STD = {moment_std:.4f}, RMSE = {moment_rmse:.4f}, R2 = {moment_r2:.4f}")


        else:
                # Concatenate all batch outputs
            pred_all = np.concatenate(preds, axis=0)     
            ground_truth_all = np.concatenate(ground_truth, axis=0)  

            # # Unnormalize the values
            pred_all_unnormalized = pred_all * dataloader.dataset.output_std.to_numpy() + dataloader.dataset.output_mean.to_numpy()
            ground_truth_all_unnormalized = ground_truth_all * dataloader.dataset.output_std.to_numpy() + dataloader.dataset.output_mean.to_numpy()

            #   # Joint Wise MAE
            abs_errors = np.abs(pred_all_unnormalized - ground_truth_all_unnormalized) 
            mae_per_joint_per_channel = abs_errors.mean(axis=0)  

            # # Joint wise RMSE
            squared_errors = (pred_all_unnormalized - ground_truth_all_unnormalized) ** 2
            rmse_per_joint_per_channel = np.sqrt(squared_errors.mean(axis=0))


            # # Standard deviation over the samples (dim=0)
            std_per_joint_per_channel = abs_errors.std(axis=0)

            # ----- R^2 per joint -----

            # Residual sum of squares (RSS)
            rss = np.sum(squared_errors, axis=0)

            # Total sum of squares (TSS)
            y_mean = np.mean(ground_truth_all_unnormalized, axis=0)
            tss = np.sum((ground_truth_all_unnormalized - y_mean) ** 2, axis=0)

            # Avoid divide-by-zero
            tss = np.clip(tss, 1e-12, None)

            r2_per_joint = 1.0 - (rss / tss)



            joints = df.columns[46:66]  # Assuming the joints are in these columns, adjust if needed 

            # Print results
            # print("----- Joint Angle Metrics -----")
            # for joint_idx, joint in enumerate(joints):
            #     if joint_idx == 10:
            #         print("----- Joint Moment Metrics -----")
            #     print(f"Joint {joint} MAE = {mae_per_joint_per_channel[joint_idx]}, STD = {std_per_joint_per_channel[joint_idx]}, RMSE = {rmse_per_joint_per_channel[joint_idx]}, R2 = {r2_per_joint[joint_idx]}")



            # ---- Split indices ----
            angle_idx = slice(0, 10)
            moment_idx = slice(10, 20)

            # ---- Aggregate metrics (mean across joints in each group) ----
            angle_mae  = mae_per_joint_per_channel[angle_idx].mean()
            angle_std  = std_per_joint_per_channel[angle_idx].mean()
            angle_rmse = rmse_per_joint_per_channel[angle_idx].mean()
            angle_r2   = 1.0 - (rss[:10].sum() / tss[:10].sum())

            moment_mae  = mae_per_joint_per_channel[moment_idx].mean()
            moment_std  = std_per_joint_per_channel[moment_idx].mean()
            moment_rmse = rmse_per_joint_per_channel[moment_idx].mean()
            moment_r2   = 1.0 - (rss[10:20].sum() / tss[10:20].sum())

            print("========================================")
            print("Joint Angle Metrics (First 10 Joints)")
            print(f"MAE = {angle_mae:.4f}, STD = {angle_std:.4f}, RMSE = {angle_rmse:.4f}, R2 = {angle_r2:.4f}")

            print("========================================")
            print("Joint Moment Metrics (Next 10 Joints)")
            print(f"MAE = {moment_mae:.4f}, STD = {moment_std:.4f}, RMSE = {moment_rmse:.4f}, R2 = {moment_r2:.4f}")

    print("PREDICTION HORIZON: ", prediction_horizon)
    
    

run_list(model_file_path, prediction_horizon=100, model_name="MANN")