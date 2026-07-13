"""
Shared helper functions used across training and evaluation scripts: moving tensors
to/from the GPU, splitting/grouping the dataset by subject or (subject, condition),
plotting predictions and PAE signal reconstructions, and computing per-joint
MAE/RMSE statistics.

Author: Chinmay Shah
Institution: Institute for Human and Machine Cognition (IHMC) / University of West Florida (UWF)
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import numpy as np
from itertools import islice


def ToDevice(x):
    """Move a tensor to the GPU if one is available, otherwise leave it on the CPU."""
    return x.cuda() if torch.cuda.is_available() else x


def Item(value):
    """Detach a tensor from the graph and move it to the CPU."""
    return value.detach().cpu()


def plot_all_columns(df: pd.DataFrame, max_cols: int = None):
    """
    Plot each column of a dataframe in its own subplot stacked vertically.
    
    Args:
        df: pandas DataFrame
        max_cols: optional limit on how many columns to plot
    """
    cols = df.columns if max_cols is None else df.columns[:max_cols]
    n = len(cols)

    fig, axes = plt.subplots(n, 1, figsize=(12, 2*n), sharex=True)

    # If only one subplot, wrap axes in a list
    if n == 1:
        axes = [axes]

    for i, col in enumerate(cols):
        axes[i].plot(df.index, df[col])
        axes[i].set_ylabel(col)
        axes[i].grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Index")

    plt.tight_layout()
    plt.show()
    

# Extract pairs from a specific data frame with a specific condition
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

# Make groups using the conditions
def make_group_dict(df: pd.DataFrame,
                    subject_col: str = "subject",
                    condition_col: str = "condition") -> Dict[Tuple[str,str], pd.DataFrame]:
    groups: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, g in df.groupby([subject_col, condition_col], sort=False):
        g = g.sort_index()
        groups[key] = g.reset_index(drop=True)
    return groups

# Splitting According to conditions or subject
def split_by_subject(df: pd.DataFrame,
                     subject_col: str = "subject",
                     train_frac=0.7, val_frac=0.15, seed=42) -> Tuple[set, set, set]:
    rng = np.random.default_rng(seed)
    subjects = sorted(df[subject_col].unique().tolist())
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(train_frac * n)
    n_val   = int(val_frac * n)
    train = set(subjects[:n_train])
    val   = set(subjects[n_train:n_train+n_val])
    # test  = set(subjects[n_train+n_val:])
    
    return train, val
    # return train, val, test

def pairs_for_subjects(pairs: List[Tuple[str,str]], subjects: set) -> List[Tuple[str,str]]:
    return [(s, c) for (s, c) in pairs if s in subjects]

def split_by_pairs(pairs: List[Tuple[str,str]], seed=42, val_frac=0.15, test_frac=0.15):
    """Optional alternative: split directly by (subject, condition) pairs."""
    rng = np.random.default_rng(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)
    n = len(pairs)
    n_val  = int(val_frac * n)
    n_test = int(test_frac * n)
    val   = set(pairs[:n_val])
    test  = set(pairs[n_val:n_val+n_test])
    train = set(pairs[n_val+n_test:])
    return train, val, test

def split_pairs_train_val(pairs: List[Tuple[str, str]],
                          val_frac: float = 0.15,
                          seed: int = 42) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    rng = np.random.default_rng(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)
    n = len(pairs)

    if n == 0:
        # nothing to split
        return [], []
    if n == 1:
        # only one group → put in train, empty val
        return pairs, []

    n_val = int(round(val_frac * n))

    # keep at least 1 in train, allow val>=1 if possible
    n_val = max(1, min(n_val, n - 1))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    return train_pairs, val_pairs


def plot_prediction(dataloader, model, col_names, tcnn=False, plot_save_name=None):
    """Run a single-step-ahead FCNN/TCNN model over `dataloader`, print per-joint MAE/STD/RMSE,
    and plot predicted vs. ground-truth trajectories for every joint channel."""
    model.eval()
    
    ground_truth = []
    preds = []
      
    with torch.no_grad():
        for batch in dataloader:
            
            
            Autoencoder_input, Predictor_input, Predictor_output = batch        
            
            Predictor_input_gpu = ToDevice(Predictor_input)
            Predictor_input_gpu_flat = Predictor_input_gpu.reshape(Predictor_input_gpu.shape[0], -1)
            
            # 1-Time Prediction
            Predictor_output = Predictor_output.squeeze(-1)  # Remove the pred_length dimension if it's 1, shape becomes [batch_size, output_features]
            
            Predictor_output_gpu = ToDevice(Predictor_output)
            
            if tcnn:
                # TCNN prediction
                y_pred = model(Predictor_input_gpu_flat)
            else:
                # FCNN prediction
                y_pred = model(Predictor_input_gpu_flat)
            
            output_np = Item(Predictor_output_gpu).numpy()
            pred_np = Item(y_pred).numpy()

            preds.append(pred_np)
            ground_truth.append(output_np)
            
            
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
    
    
    joints = col_names 

    # Print results
    for joint_idx, joint in enumerate(joints):
        print(f"Joint {joint} MAE = {mae_per_joint_per_channel[joint_idx]}, STD = {std_per_joint_per_channel[joint_idx]}, RMSE = {rmse_per_joint_per_channel[joint_idx]} ")
        
    # Assume pred and ground_truth are (N, 20)
    N, num_features = pred_all_unnormalized.shape

    rows = num_features // 2   # 10 rows if 20 features
    cols = 2
    
    # Number of features you want in each figure
    features_per_plot = 10  

    # Loop over feature groups
    for group in range((num_features + features_per_plot - 1) // features_per_plot):
        start = group * features_per_plot
        end = min((group + 1) * features_per_plot, num_features)

        # Create a grid: here 2 rows x 5 cols for 10 features
        rows, cols = 2, 5
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, feat_idx in enumerate(range(start, end)):
            ax = axes[i]
            ax.plot(pred_all_unnormalized[:, feat_idx], label="Prediction", linewidth=1.2, color="red")
            ax.plot(ground_truth_all_unnormalized[:, feat_idx], label="Ground Truth", linewidth=1.0, color="black", alpha=0.7)
            
            # Use joint name instead of generic index
            ax.set_title(joints[feat_idx])

            if feat_idx == 0:  # only first subplot gets the legend
                ax.legend(loc="upper right")

        # Hide any unused subplots in the last group
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.xlabel("Time (samples)")
        plt.tight_layout()
    
    
# Plot Training vs Validation Loss Curves
def plot_train_val_loss(train_losses, val_losses):
    """
    Plots training and validation loss curves.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    
def plot_PAE_recon(dataloader, model, seq_length, col_names):
    """Plot the PAE model's reconstructed signal against the input IMU signal, per channel."""
    model.eval()
    fig, axs = plt.subplots(3, 7, figsize=(30,10), sharex=True, sharey=True)
    
    step = 50
    stop_plotting = 1
    
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            # start_index = j * step
            # end_index = start_index + seq_length
            
            # Unpack batch
            input_tensor, _, _ = batch
            input_tensor = ToDevice(input_tensor)  # your helper to put tensor on GPU/CPU

            # Model prediction
            output, _, _, _ = model(input_tensor)
            
            # Convert to numpy after flattening batch & time together
            # output shape: [batch, 21, 201] -> [21, batch*201]
            output_flat = output.permute(1, 0, 2).reshape(21, -1).cpu().numpy()
            input_flat = input_tensor.permute(1, 0, 2).reshape(21, -1).cpu().numpy()
            
            
    
            feature_len = output_flat.shape[1]
            # Flattened time index for plotting
            time_idx = range(k*feature_len, (k+1)*feature_len)
            
            for j in range(0, output_flat.shape[1], step):
                
                start_index = j
                end_index = min(start_index + seq_length, feature_len)
                
                for i in range(output_flat.shape[0]):
                    row = i % 3
                    col = i // 3
                    ax = axs[row, col]

                    # Plot predicted and ground truth
                    ax.plot(time_idx[start_index:end_index], output_flat[i, start_index:end_index], linewidth=1, alpha=0.75)
                    ax.plot(time_idx[start_index:end_index], input_flat[i, start_index:end_index], linewidth=1, color='black')

                    # Title and ticks
                    ax.set_title(col_names[i])
                    ax.tick_params(labelsize=8)
                    
            if k==stop_plotting:
                break  # Only plot the first batch for visualization
            
    plt.tight_layout()
    # plt.show()
    

def plot_MANN_predictions(dataloader, PAE_model, model, col_names, tcnn=False):
    """Feed IMU windows through the PAE model to get phase parameters, use them to gate the
    MANN/MoE-TCN predictor, then plot predicted vs. ground-truth joint trajectories."""
    model.eval()
        
    ground_truth = []
    preds = []
    
    
    with torch.no_grad():
        for batch in dataloader:
            
            Autoencoder_inputs, Predictor_input, Predictor_output = batch

            Autoencoder_inputs = ToDevice(Autoencoder_inputs)
            
            PAE_model.eval()
            params  = PAE_model(Autoencoder_inputs)
            
            
            params_cat = torch.cat(params, dim=2)
            phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
            phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
            phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])
            
            phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 

            phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)
            
            # Flattening the inputs for the motion prediction network
            # flattened_inputs = utility.ToDevice(Predictor_input.reshape(Predictor_input.shape[0], -1))
            
            # Only using the last 20 time steps to predict the future time step
            last_step_inputs = Predictor_input[:, :, -50:]              # shape = [batch, features]
            flattened_inputs = ToDevice(last_step_inputs.reshape(last_step_inputs.shape[0], -1)) # already flat
            
            # FCNN_combine_inputs = torch.cat((flattened_inputs, phaseInputs), dim=1)
            
            # MANN - 1 Step in the future prediction
            if tcnn:
                y_pred = model(phaseInputs, ToDevice(last_step_inputs))
            else:
                y_pred, _ = model(phaseInputs, flattened_inputs) 
            
            output_np = Predictor_output.squeeze(-1).numpy()
            pred_np = Item(y_pred).numpy()

            preds.append(pred_np)
            ground_truth.append(output_np)
                  

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
    
    
    joints = col_names 

    # Print results
    for joint_idx, joint in enumerate(joints):
        print(f"Joint {joint} MAE = {mae_per_joint_per_channel[joint_idx]}, STD = {std_per_joint_per_channel[joint_idx]}, RMSE = {rmse_per_joint_per_channel[joint_idx]} ")
        
    # Assume pred and ground_truth are (N, 20)
    N, num_features = pred_all_unnormalized.shape

    rows = num_features // 2   # 10 rows if 20 features
    cols = 2
    
    # Number of features you want in each figure
    features_per_plot = 10  

    # Loop over feature groups
    for group in range((num_features + features_per_plot - 1) // features_per_plot):
        start = group * features_per_plot
        end = min((group + 1) * features_per_plot, num_features)

        # Create a grid: here 2 rows x 5 cols for 10 features
        rows, cols = 2, 5
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, feat_idx in enumerate(range(start, end)):
            ax = axes[i]
            ax.plot(pred_all_unnormalized[:, feat_idx], label="Prediction", linewidth=1.2, color="red")
            ax.plot(ground_truth_all_unnormalized[:, feat_idx], label="Ground Truth", linewidth=1.0, color="black", alpha=0.7)
            
            # Use joint name instead of generic index
            ax.set_title(joints[feat_idx])

            if feat_idx == 0:  # only first subplot gets the legend
                ax.legend(loc="upper right")

        # Hide any unused subplots in the last group
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.xlabel("Time (samples)")
        plt.tight_layout()
        # plt.show()
        
        
def plot_predictor_results(dataloader, model, col_names, window, PAE_model=None, moe_tcnn=False, mann=False, fcnn_sw=False):
    """Plot predicted vs. ground-truth joint angle/moment trajectories for any of the
    supported predictor models (MoE-TCN, MANN, FCNN sliding-window, or plain TCNN)."""
    model.eval()
    fig1, axs1 = plt.subplots(2, 5, figsize=(30,10), sharey=True, sharex=True)
    fig2, axs2 = plt.subplots(2, 5, figsize=(30,10), sharey=True, sharex=True)
    
    axs1_flat = axs1.flatten()
    axs2_flat = axs2.flatten()
    
    step = 1
    if window > 1:
        step = 10
    stop_plotting = 1
    
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
               
            Autoencoder_inputs, Predictor_inputs, Predictor_outputs = batch

            Autoencoder_inputs = ToDevice(Autoencoder_inputs)
            Predictor_input_gpu = ToDevice(Predictor_inputs)
            
            # 1-Time Prediction
            Predictor_outputs = Predictor_outputs.squeeze(-1)  # Remove the pred_length dimension if it's 1, shape becomes [batch_size, output_features]
            
            if moe_tcnn:
                PAE_model.eval()
                params  = PAE_model(Autoencoder_inputs)

                params_cat = torch.cat(params, dim=2)
                phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
                phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
                phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])

                phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 

                phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)

                # Flattening the inputs for the motion prediction network
                # flattened_inputs = utility.ToDevice(Predictor_input.reshape(Predictor_input.shape[0], -1))

                # Only using the last 20 time steps to predict the future time step
                last_step_inputs = Predictor_input_gpu[:, :, -50:]              # shape = [batch, features]
                flattened_inputs = ToDevice(last_step_inputs.reshape(last_step_inputs.shape[0], -1)) # already flat

                # FCNN_combine_inputs = torch.cat((flattened_inputs, phaseInputs), dim=1)
                
                y_pred = model(phaseInputs, last_step_inputs)
            
            elif mann:
                PAE_model.eval()
                params  = PAE_model(Autoencoder_inputs)

                params_cat = torch.cat(params, dim=2)
                phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
                phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
                phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])

                phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 

                phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)

                # Flattening the inputs for the motion prediction network
                # flattened_inputs = utility.ToDevice(Predictor_input.reshape(Predictor_input.shape[0], -1))

                # Only using the last 20 time steps to predict the future time step
                last_step_inputs = Predictor_input_gpu[:, :, -50:]              # shape = [batch, features]
                flattened_inputs = ToDevice(last_step_inputs.reshape(last_step_inputs.shape[0], -1)) # already flat

                # FCNN_combine_inputs = torch.cat((flattened_inputs, phaseInputs), dim=1)
                
                y_pred, _ = model(phaseInputs, flattened_inputs)
            elif fcnn_sw:
                Predictor_input_gpu_flat = Predictor_input_gpu.reshape(Predictor_input_gpu.shape[0], -1)
                # FCNN sliding window prediction
                y_pred = model(Predictor_input_gpu_flat)        
            else:
                # TCNN 1 Step prediction
                y_pred = model(Predictor_input_gpu)
                                
            
            
            pred_np = Item(y_pred).permute(1, 0, 2).reshape(20,-1).numpy()
            ground_truth_np = Predictor_outputs.permute(1, 0, 2).reshape(20,-1).numpy()
            
            batch_len = pred_np.shape[0]
            feature_len = pred_np.shape[1]
            
            time_idx = range(j*feature_len, (j+1)*feature_len)
            
            
            for i in range(0, feature_len, step):
                
                start_index = i
                # end_index = start_index + dataloader.dataset.pred_len
                end_index = min(start_index + window, feature_len)
                
                
                for k in range(10):
                    
                    
                    axs1_flat[k].plot(time_idx[start_index:end_index], pred_np[k,start_index:end_index], linewidth=1, alpha=1.0, color="red", label="Pred")
                    axs1_flat[k].plot(time_idx[start_index:end_index], ground_truth_np[k,start_index:end_index], linewidth=1, alpha=0.75, color="black", label="Ground Truth")
                    axs1_flat[k].set_title(col_names[k])
                    
                    axs2_flat[k].plot(time_idx[start_index:end_index], pred_np[k+10,start_index:end_index], linewidth=1, alpha=1.0, color="red", label="Pred")
                    axs2_flat[k].plot(time_idx[start_index:end_index], ground_truth_np[k+10,start_index:end_index], linewidth=1, alpha=0.75, color="black", label="Ground Truth")
                    axs2_flat[k].set_title(col_names[k+10])
                    
                    
                
    
            if j == stop_plotting:
                break
                         
    plt.tight_layout()
    plt.show()            
                
def stats_predictor_cal(dataloader, model, col_names, PAE_model=None, moe_tcnn=False, mann=False, fcnn_sw=False):
    """Run any of the supported predictor models over `dataloader` and print per-joint
    MAE, STD, and RMSE (in original, unnormalized units) accumulated over the full multi-step horizon."""
    model.eval()
    
    out_std = torch.as_tensor(dataloader.dataset.output_std, dtype=torch.float64)  # [F]

    # Running accumulators (float64 for stability)
    sum_abs   = torch.zeros_like(out_std, dtype=torch.float64)  # Σ |e|
    sum_abs2  = torch.zeros_like(out_std, dtype=torch.float64)  # Σ |e|^2  (for std of abs error)
    sum_sq    = torch.zeros_like(out_std, dtype=torch.float64)  # Σ e^2    (for RMSE)
    count     = torch.tensor(0, dtype=torch.int64)  # total samples per feature


    with torch.no_grad():
        for batch in dataloader:
            
            Autoencoder_inputs, Predictor_inputs, Predictor_outputs = batch

            Autoencoder_inputs = ToDevice(Autoencoder_inputs)
            Predictor_input_gpu = ToDevice(Predictor_inputs)
            
            # 1-Time Prediction
            Predictor_outputs = Predictor_outputs.squeeze(-1)  # Remove the pred_length dimension if it's 1, shape becomes [batch_size, output_features]
            
            
            if moe_tcnn:
                PAE_model.eval()
                params  = PAE_model(Autoencoder_inputs)

                params_cat = torch.cat(params, dim=2)
                phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
                phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
                phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])

                phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 

                phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)

                # Flattening the inputs for the motion prediction network
                # flattened_inputs = utility.ToDevice(Predictor_input.reshape(Predictor_input.shape[0], -1))

                # Only using the last 20 time steps to predict the future time step
                last_step_inputs = Predictor_input_gpu[:, :, -50:]              # shape = [batch, features]
                flattened_inputs = ToDevice(last_step_inputs.reshape(last_step_inputs.shape[0], -1)) # already flat

                # FCNN_combine_inputs = torch.cat((flattened_inputs, phaseInputs), dim=1)
                
                y_pred = model(phaseInputs, last_step_inputs)
                
            elif mann:
                
                PAE_model.eval()
                params  = PAE_model(Autoencoder_inputs)

                params_cat = torch.cat(params, dim=2)
                phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
                phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
                phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])

                phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 

                phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)

                # Flattening the inputs for the motion prediction network
                # flattened_inputs = utility.ToDevice(Predictor_input.reshape(Predictor_input.shape[0], -1))

                # Only using the last 20 time steps to predict the future time step
                last_step_inputs = Predictor_input_gpu[:, :, -50:]              # shape = [batch, features]
                flattened_inputs = ToDevice(last_step_inputs.reshape(last_step_inputs.shape[0], -1)) # already flat


                y_pred, _ = model(phaseInputs, flattened_inputs)
            elif fcnn_sw:
                Predictor_input_gpu_flat = Predictor_input_gpu.reshape(Predictor_input_gpu.shape[0], -1)
                # FCNN sliding window prediction
                y_pred = model(Predictor_input_gpu_flat)    
            else:
                # TCNN 1 Step prediction
                y_pred = model(Predictor_input_gpu)

            # errors in ORIGINAL scale: (pred - gt) * std
            # broadcast std over [B, F, T]
                        
            err = (Item(y_pred) - Predictor_outputs).to(dtype=torch.float64) * out_std.view(1, -1, 1)

            abs_err = err.abs()                       # [B, F, T]
            sq_err  = err.pow(2)                      # [B, F, T]
            abs_err2 = abs_err.pow(2)

            # reduce over batch & time -> per-feature vectors
            reduce_dims = (0, 2)
            sum_abs   += abs_err.sum(dim=reduce_dims)
            sum_abs2  += abs_err2.sum(dim=reduce_dims)
            sum_sq    += sq_err.sum(dim=reduce_dims)
            count     += err.shape[0] * err.shape[2]  # B*T

    # finalize metrics
    count_f = count.to(torch.float64).clamp_min(1)
    mae  = (sum_abs / count_f)                                 # per-feature
    rmse = torch.sqrt(sum_sq / count_f)                         # per-feature
    # std of absolute error (like np.std(|e|))
    mean_abs = mae
    var_abs  = (sum_abs2 / count_f) - mean_abs.pow(2)
    var_abs  = torch.clamp(var_abs, min=0.0)
    std_abs  = torch.sqrt(var_abs)

    # print nicely
    mae_np  = mae.cpu().numpy()
    std_np  = std_abs.cpu().numpy()
    rmse_np = rmse.cpu().numpy()

    print("Samples per feature (total N per feature):", int(count.item()))
    for i, joint in enumerate(col_names):
        print(f"Joint {joint} MAE = {mae_np[i]:.4f}, STD = {std_np[i]:.4f}, RMSE = {rmse_np[i]:.4f}")        

