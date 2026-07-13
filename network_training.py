"""
Main training entry point for the sensor-suit motion prediction models.

`main()` selects one of the supported architectures via the `model_to_train` flag
("PAE", "MANN", "MoETCNN", "TCNN", "FCNN_SW", "LSTM"), builds the train/validation
dataloaders from the combined dataset CSV (see DataLoader/data_loader_pae.py),
trains the model (optionally logging to Weights & Biases), saves the trained
weights + config to `Saved Models/`, and generates evaluation plots/statistics.

Update `data_path`, `pae_model_file_path`, and `model_file_path` in `main()` to
point at your local dataset/checkpoint locations before running (see README).

Author: Chinmay Shah
Institution: Institute for Human and Machine Cognition (IHMC) / University of West Florida (UWF)
"""

from datetime import datetime
import json
import os
import wandb
import pandas as pd
from Library import utility
from DataLoader.data_loader_pae import GroupedBatchSampler, GroupedSequenceDataset
from torch.utils.data import Dataset, DataLoader, Subset
from Models.TCNN import TCNModel, TCNModel_Forecast
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Models import PAE
from Models.MANN import Model
from Models.TCNN_MOE import MANN_TCN_DynamicWeights, MANN_TCN_DynamicWeights_Forecast
from Models.FCNN import FCNN
from Models.LSTM import LSTM

# Repo root (folder this file lives in) - used so the default paths below work
# regardless of where the repository is cloned.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_predictor_model(model, config, training_dataloader, validation_dataloader, tcnn=False, log_wandB=False):
    """Train a direct motion predictor (TCNN, FCNN_SW, or LSTM) with Adam + MSE loss,
    validating each epoch and optionally logging to Weights & Biases."""

    ## Setting up an optimizer and a loss function - Original Paper used a AdamWr optimizer We using a simple SGD
    learning_rate = config["lr"]
    momentum = config["momentum"]

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss function
    lossFn = nn.MSELoss()
    # lossFn_no_reduction = nn.MSELoss(reduction='none')
    
    for batch in training_dataloader:

        Autoencoder_input, Predictor_input, Predictor_output = batch
        print("Autoencoder_input Shape: ", Autoencoder_input.shape)
        print("Predictor_input Shape: ", Predictor_input.shape)
        print("Predictor_output Shape: ", Predictor_output.shape)
        break
    
    ## Training the periodic auto encoder
    print("Starting Training........")
    training_losses = []
    validation_losses = []

    epochs = config["epochs"]
    
    ## Training Loop
    for epoch in range(epochs):
        model.train()
    
        running_loss = 0.0
    
        for batch in training_dataloader:
        
            Autoencoder_input, Predictor_input, Predictor_output = batch        
            
            Predictor_input_gpu = utility.ToDevice(Predictor_input)
            Predictor_input_gpu_flat = Predictor_input_gpu.reshape(Predictor_input_gpu.shape[0], -1)
            
            # 1-Time Prediction (It only works if there is a row with 1)
            Predictor_output = Predictor_output.squeeze(-1)  # Remove the pred_length dimension if it's 1, shape becomes [batch_size, output_features]
            
            Predictor_output_gpu = utility.ToDevice(Predictor_output)  
            
            if tcnn:
                # TCNN Prediction
                y_pred = model(Predictor_input_gpu)
            else:
                
                # FCNN Prediction
                y_pred = model(Predictor_input_gpu_flat)
                
            # Calculate the loss            
            loss = lossFn(y_pred, Predictor_output_gpu)

            # # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print statistics
            running_loss += loss.item() * Predictor_input.size(0)
                
        train_loss = running_loss / len(training_dataloader.dataset)
        training_losses.append(train_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss}')
        
        val_loss = calc_val_loss(model, validation_dataloader, lossFn, tcnn=tcnn)
        validation_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
        
        if log_wandB:
            wandb.log({"train/train_loss": train_loss,
                        "train/epoch": epoch,
                        "val/val_loss": val_loss,
                        "val/epoch":epoch})
        
            
    return training_losses, validation_losses  

  
def calc_val_loss(model, validation_dataloader, lossFn, tcnn=False, lossFn_no_reduction=None):
    """Compute mean validation loss for a direct motion predictor (TCNN/FCNN_SW/LSTM)."""
    model.eval()
    
    val_loss = 0.0
    # individual_losses = np.zeros(6, dtype=np.float32)
    
    with torch.no_grad():
        for batch in validation_dataloader:
            
            Autoencoder_input, Predictor_input, Predictor_output = batch    
            Predictor_input_gpu = utility.ToDevice(Predictor_input)
            Predictor_input_gpu_flat = Predictor_input_gpu.reshape(Predictor_input_gpu.shape[0], -1)
                        
            # 1-Time Prediction
            Predictor_output = Predictor_output.squeeze(-1)  # Remove the pred_length dimension if it's 1, shape becomes [batch_size, output_features]
            
            Predictor_output_gpu = utility.ToDevice(Predictor_output)
            
            if tcnn:
                # TCNN Prediction
                y_pred = model(Predictor_input_gpu)
            else:
                
                # FCNN Prediction
                y_pred = model(Predictor_input_gpu_flat)
            
            # Calculate the loss
            loss = lossFn(y_pred, Predictor_output_gpu)

            
            # Calculate running loss
            val_loss += loss.item() * Predictor_input.size(0)
            
            
        val_loss = val_loss / len(validation_dataloader.dataset)

    return val_loss

def train_PAE_model(model, config, training_dataloader, validation_dataloader, log_wandB=False):
    """Train the Periodic Autoencoder (PAE) as a signal-reconstruction task with Adam + MSE loss."""

    ## Setting up an optimizer and a loss function - Original Paper used a AdamWr optimizer We using a simple SGD
    learning_rate = config["lr"]
    
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss function
    lossFn = nn.MSELoss()
    lossFn_no_reduction = nn.MSELoss(reduction='none')

    ## Training the periodic auto encoder
    print("Starting Training........")
    training_losses = []
    validation_losses = []
    individual_losses = []

    epochs = config["epochs"]
    
    # Input data shape
    for batch in training_dataloader:
        print(batch[0].shape)
        break
    
    ## Training Loop
    for epoch in range(epochs):
    
        running_loss = 0.0
    
        for batch in training_dataloader:
        
            Autoencoder_inputs, _, _ = batch
            Autoencoder_inputs = utility.ToDevice(Autoencoder_inputs)

            # Zero the parameter gradients
            optimizer.zero_grad()
        
            # Forward
            outputs, latent, signal, params  = model(Autoencoder_inputs)
        
            # Flattening the outputs and inputs and calculating the loss
            # flattened_inputs = Autoencoder_inputs.reshape(Autoencoder_inputs.shape[0], -1)
            # flattened_outputs = outputs.reshape(outputs.shape[0], -1)
        
            # Calculate Loss
            loss = lossFn(Autoencoder_inputs, outputs)
        
            # Backward
            loss.backward()
            optimizer.step()
        
            # Calculate running loss
            running_loss += loss.item() * Autoencoder_inputs.size(0)
                                             
        train_loss = running_loss / len(training_dataloader.dataset)
        training_losses.append(train_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss}')
    
    
        val_loss = cal_PAE_val_loss(model, validation_dataloader, lossFn, lossFn_no_reduction)
        validation_losses.append(val_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
        
        
        if log_wandB:
            wandb.log({"train/train_loss": train_loss,
                        "train/epoch": epoch,
                        "val/val_loss": val_loss,
                        "val/epoch":epoch})
            
    print('Finished Training')
    
    return training_losses, validation_losses

def cal_PAE_val_loss(model, validation_dataloader, lossFn, lossFn_no_reduction):
    """Compute mean PAE reconstruction validation loss."""
    model.eval()
    
    val_loss = 0.0
    # individual_losses = np.zeros(21, dtype=np.float32)
    
    # Convoluted Signal
    all_latents = []
    
    # Reconstructed signal
    all_signals = []
    
    with torch.no_grad():
        for batch in validation_dataloader:
            
            Autoencoder_inputs, _, _ = batch
            Autoencoder_inputs = utility.ToDevice(Autoencoder_inputs)
                        
            # Predict
            # outputs, latent, signal, params  = model(PAE_inputs)
            outputs, _, _, _  = model(Autoencoder_inputs)
                    
            # Flattening the outputs and inputs and calculating the loss
            # flattened_inputs = Autoencoder_inputs.reshape(Autoencoder_inputs.shape[0], -1)
            # flattened_outputs = outputs.reshape(outputs.shape[0], -1)
        
            # Calculate Total Loss
            loss = lossFn(Autoencoder_inputs, outputs)
            
            # Calculate running loss
            val_loss += loss.item() * Autoencoder_inputs.size(0)
            
            # ensure no lingering references
            del outputs, loss
            
            # Append all the latents and signals
            # all_latents.append(utility.Item(latent))
            # all_signals.append(utility.Item(signal))  
            
        
        val_loss = val_loss / len(validation_dataloader.dataset)
        # individual_losses = individual_losses / len(validation_dataloader.dataset)   
        
        
    return val_loss

def train_MANN_model(model, config, training_dataloader, validation_dataloader, PAE_model, tcnn=False, log_wandB=False, collect_phase=False):
    """Train the MANN or MoE-TCN predictor: freezes a pretrained PAE_model to derive
    phase-based gating inputs each step, then trains `model` with Adam + MSE loss.
    If `collect_phase` is set, only accumulates phase-feature statistics and returns early
    (used to compute normalization stats, not for actual training)."""

    if collect_phase:
        all_phases = []
        
    ## Setting up an optimizer and a loss function - Original Paper used a AdamWr optimizer We using a simple SGD
    learning_rate = config["lr"]

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss function
    lossFn = nn.MSELoss()
    # lossFn_no_reduction = nn.MSELoss(reduction='none')
    
    for batch in training_dataloader:

        Autoencoder_input, Predictor_input, Predictor_output = batch
        print("Autoencoder_input Shape: ", Autoencoder_input.shape)
        print("Predictor_input Shape: ", Predictor_input.shape)
        print("Predictor_output Shape: ", Predictor_output.shape)
        break

    ## Training the periodic auto encoder
    print("Starting Training........")
    training_losses = []
    validation_losses = []

    epochs = config["epochs"]
    
    
    ## Training Loop
    for epoch in range(epochs):
    
        running_loss = 0.0
    
        for batch in training_dataloader:
        
            Autoencoder_input, Predictor_input, Predictor_output = batch
            
            Autoencoder_input = utility.ToDevice(Autoencoder_input)
            
            # 1-Time Prediction
            Predictor_output = Predictor_output.squeeze(-1)
            
            PAE_model.eval()
            # _, _, _, params  = PAE_model(Autoencoder_input)
            params  = PAE_model(Autoencoder_input)
            
            
            params_cat = torch.cat(params, dim=2)
            phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
            phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
            phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])
            
            phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 

            phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)
            
            if collect_phase:
                # Select features 20:50 (Python indexing 20:50)
                phase_20_50 = phaseInputs[:, 20:50]  # (B, 30)
                # Flatten across batch: (B,30) -> (B*30,)
                all_phases.append(phase_20_50.detach().cpu().numpy())
            
                continue
                
            
            # Flattening the inputs for the motion prediction network
            # flattened_inputs = utility.ToDevice(Predictor_input.reshape(Predictor_input.shape[0], -1))
            
            # Only using the last 20 time steps to predict the future time step
            last_step_inputs = Predictor_input[:, :, -50:]              # shape = [batch, features]
            flattened_inputs = utility.ToDevice(last_step_inputs.reshape(last_step_inputs.shape[0], -1)) # already flat
            
            # FCNN_combine_inputs = torch.cat((flattened_inputs, phaseInputs), dim=1)
            
            # MANN - 1 Step in the future prediction
            
            if tcnn:
                y_pred = model(phaseInputs, utility.ToDevice(last_step_inputs))
            else:
                y_pred, _ = model(phaseInputs, flattened_inputs) 
                            
                            
            # Calculate the loss
            loss = lossFn(y_pred, utility.ToDevice(Predictor_output))

            # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
      
            # Print statistics
            running_loss += loss.item() * Predictor_input.size(0)    
        
        
        if collect_phase and len(all_phases) > 0:
            all_phase_arr = np.concatenate(all_phases, axis=0)  # (total_samples*30,)
            phase_mean = all_phase_arr.mean(axis=0)
            phase_std  = all_phase_arr.std(axis=0)
            print("Collected phase data across batches")
            print("Phase 20:50 mean:", phase_mean)
            print("Phase 20:50 std:", phase_std)
            
            return 0, 0
        
        
        
        train_loss = running_loss / len(training_dataloader.dataset)
        training_losses.append(train_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss}')
        
        val_loss = calc_MANN_val_loss(model, PAE_model, validation_dataloader, lossFn, tcnn)
        validation_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
        
        if log_wandB:
            wandb.log({"train/train_loss": train_loss,
                        "train/epoch": epoch,
                        "val/val_loss": val_loss,
                        "val/epoch":epoch})
        

            
    return training_losses, validation_losses  

def calc_MANN_val_loss(model, PAE_model, validation_dataloader, lossFn, tcnn=False, lossFn_no_reduction=None):
    """Compute mean validation loss for the MANN/MoE-TCN predictor, gated by the frozen PAE_model."""
    model.eval()
    PAE_model.eval()
    
    val_loss = 0.0
    # individual_losses = np.zeros(6, dtype=np.float32)
    
    with torch.no_grad():
        for batch in validation_dataloader: 
        
            Autoencoder_input, Predictor_input, Predictor_output = batch
            
            Autoencoder_input = utility.ToDevice(Autoencoder_input)
            
            # 1-Time Prediction
            Predictor_output = Predictor_output.squeeze(-1)
            
            PAE_model.eval()
            # _, _, _, params  = PAE_model(Autoencoder_input)
            params  = PAE_model(Autoencoder_input)
            
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
            flattened_inputs = utility.ToDevice(last_step_inputs.reshape(last_step_inputs.shape[0], -1)) # already flat
            
            # FCNN_combine_inputs = torch.cat((flattened_inputs, phaseInputs), dim=1)
            
            # MANN - 1 Step in the future prediction
            if tcnn:
                y_pred = model(phaseInputs, utility.ToDevice(last_step_inputs))
            else:
                y_pred, _ = model(phaseInputs, flattened_inputs) 
                        
            # Calculate the loss
            loss = lossFn(y_pred, utility.ToDevice(Predictor_output))
            
            # Calculate running loss
            val_loss += loss.item() * Predictor_input.size(0)
     

        val_loss = val_loss / len(validation_dataloader.dataset)

    return val_loss

def main():
    """Configure, build, train (or load), and evaluate one motion-prediction model.

    Edit the flags/paths below to choose the model, prediction horizon, and dataset
    location, then run this file directly: `python network_training.py`.
    """
    # Logging Flag
    # Plot Flag
    log_wandB = True
    train_model_flag = True
    save_file = True
    future_forcast = False
    collect_phase = False
    
    time_horizon_prediction = 100 # Can be 1, 5, 20, 50, 80, 100
    
    if time_horizon_prediction != 1:
        future_forcast = True
    
    model_to_train = "MoETCNN" # Can be "MANN", "MoETCNN", "PAE", "RNN" 
    
    if save_file:
        print("Saving the model after training !!!")
    
    # Data Setup
    # TODO: add the path to your combined training CSV here (produced by running
    # data_extraction.py on your local copy of the dataset - see README).
    data_path = os.path.join(BASE_DIR, "Data - Second Skin", "Testing", "<your_dataset>.csv")

    # Model File to load
    # TODO: replace this filename with your own trained PAE checkpoint (only needed
    # for model_to_train = "MANN" / "MoETCNN", which gate on a pretrained PAE).
    pae_model_file_path = os.path.join(BASE_DIR, "Saved Models", "20260212_0245_PAE on SS Dataset - 10 Subjects - 200 epochs (256 batch size).pth")
    # TODO: replace this filename with your own trained checkpoint (only used when
    # train_model_flag = False, to load a model instead of training one).
    model_file_path = os.path.join(BASE_DIR, "Saved Models", "20260219_1302_MANN model for robot deployment .pth")
    
    file_name = model_to_train +" for final Paper test k = " + str(time_horizon_prediction)
    project_name = "IROS-2026-Second-Skin-Dataset-Project"
    
    # Config the configurations
    config = {
        "training_tag": file_name,
        "project_name": project_name,
        "epochs": 1,
        "batch_size": 128,
        "num_workers": 8,
        "momentum":0.9,
        "lr": 1e-4,
        "dataset": "IHMC Senorsuit",
        "seq_length": 201,
        "pred_length": time_horizon_prediction,
    }

    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        project_name = config["project_name"]
        wandb.init( project=project_name, name= config["training_tag"], config=config)
          
    # Setup Data Frames
    df = pd.read_csv(data_path)
    print("Dataset Size: ", df.shape)
    
    # Extract Windows randomly across different trials and conditions and subjects
    subjects, conditions, all_pairs = utility.extract_pairs(df, "subject", "condition")
    groups = utility.make_group_dict(df, "subject", "condition")
    
    # Split for training and validation
    train_pairs, val_pairs = utility.split_pairs_train_val(all_pairs, val_frac=0.15)
        
    train_ds = GroupedSequenceDataset(
        df, seq_len=config["seq_length"], pred_len=config["pred_length"], stride=1,
        subject_col="subject", condition_col="condition",
        include_groups=train_pairs
        )
    
    val_ds = GroupedSequenceDataset(
        df, seq_len=config["seq_length"], pred_len=config["pred_length"], stride=1,
        subject_col="subject", condition_col="condition",
        include_groups=val_pairs
        )

    # Grouped batch samplers to ensure windows from the same (subject, condition) are in the same batch
    train_sampler = GroupedBatchSampler(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=False)
    train_loader  = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=16, pin_memory=True)
    
    val_sampler = GroupedBatchSampler(val_ds, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    val_loader  = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=16, pin_memory=True)
    
    train_sampler_plot = GroupedBatchSampler(train_ds, batch_size=64, shuffle=False, drop_last=False)
    train_loader_plot  = DataLoader(train_ds, batch_sampler=train_sampler_plot, num_workers=8, pin_memory=True)
    
    val_sampler_plot = GroupedBatchSampler(val_ds, batch_size=64, shuffle=False, drop_last=False)
    val_loader_plot  = DataLoader(val_ds, batch_sampler=val_sampler_plot, num_workers=8, pin_memory=True)
    
    
    
    # Model setup
    # Load the model structure - then either train or populate from a file
    
    if model_to_train == "TCNN":
        
        
        if future_forcast:
            model = utility.ToDevice(TCNModel_Forecast(
                                        input_size=46,
                                        output_size=20,
                                        horizon=time_horizon_prediction,
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

    elif model_to_train == "PAE":
        # input_channels must match the PAE_input width sliced in
        # DataLoader/data_loader_pae.py (g.iloc[:, 21:42] -> 21 channels), and
        # intermediate_channels=16 matches the "MANN"/"MoETCNN" PAE config below
        # so a single trained PAE checkpoint can be reused across all three.
        model = utility.ToDevice(PAE.Model(
                              input_channels=21,
                              embedding_channels=10,
                              intermediate_channels=16,
                              time_range=201,
                              window=1.0
                             ))
        
    elif model_to_train == "FCNN_SW":
            
        model = utility.ToDevice(FCNN(
                                    inputs=46, 
                                    outputs=20, 
                                    numOfLayers=5, 
                                    hiddenDimension=512, 
                                    input_seq_len=201, 
                                    predictionHorizon=time_horizon_prediction, 
                                    dropoutRate=0.2))
            
    elif model_to_train == "MANN":
            
        ## Load PAE file
        weights = torch.load(pae_model_file_path, weights_only=True)
        PAE_model = utility.ToDevice(PAE.Model(
                          input_channels=21,
                          embedding_channels=10,
                          intermediate_channels=16,
                          time_range=201,
                          window=1.0
                         ))
    
        PAE_model.load_state_dict(weights)
        
        # Setting up a mode adaptive neural network MANN
        model = utility.ToDevice(Model( gating_input=50,
                                               gating_hidden=256,
                                               gating_output=10,
                                               main_input=46*50,
                                               main_hidden=256,
                                               main_output=20,
                                               prediction_horizon=time_horizon_prediction,
                                               dropout=0.2))
            
    elif model_to_train == "MoETCNN":
            
                        
        ## Load PAE file
        # input_channels/intermediate_channels must match the checkpoint at
        # pae_model_file_path (same PAE config as the "MANN" branch above -
        # see the "PAE" branch for why 21/16 is correct).
        weights = torch.load(pae_model_file_path, weights_only=True)
        PAE_model = utility.ToDevice(PAE.Model(
                          input_channels=21,
                          embedding_channels=10,
                          intermediate_channels=16,
                          time_range=201,
                          window=1.0
                         ))

        PAE_model.load_state_dict(weights)

        if future_forcast:

            model = utility.ToDevice(MANN_TCN_DynamicWeights_Forecast(
                                        input_size=46,
                                        output_size=20,
                                        horizon=time_horizon_prediction,
                                        num_experts=10,
                                        tcn_channels=[64, 128, 128, 256, 64],
                                        gating_input=50,
                                        gating_hidden=256,
                                        kernel_size=6,
                                        tcn_dropout=0.2,
                                        gating_dropout=0.2))
            
            
        else:    
            model = utility.ToDevice(MANN_TCN_DynamicWeights( 
                                        input_size=46,
                                        output_size=20,
                                        num_experts=10,
                                        tcn_channels=[64, 128, 128, 256, 64],
                                        gating_input=50,
                                        gating_hidden=256,
                                        kernel_size=6,
                                        tcn_dropout=0.2,
                                        gating_dropout=0.2))
            
    elif model_to_train == "LSTM":
            
            model = utility.ToDevice(LSTM(input_size=46,
                                          hidden_size=256,
                                          num_layers=3,
                                          output_size=20,
                                          pred_horizon=time_horizon_prediction,
                                          dropout=0.4))
            
    if train_model_flag:
        
        
        if model_to_train == "TCNN":
            
            # Train
            train_loss, val_loss = train_predictor_model(model=model, config=config, training_dataloader=train_loader, 
                                                       validation_dataloader=val_loader, tcnn=True, log_wandB=log_wandB)
        
        elif model_to_train == "PAE":
            
            # Train
            train_loss, val_loss = train_PAE_model(model=model, config=config, training_dataloader=train_loader, 
                                                       validation_dataloader=val_loader, log_wandB=log_wandB)
        
        elif model_to_train == "FCNN_SW":
            
            # Train
            train_loss, val_loss = train_predictor_model(model=model, config=config, training_dataloader=train_loader, 
                                                       validation_dataloader=val_loader, log_wandB=log_wandB)
            
        elif model_to_train == "MANN":
            
            # Train
            train_loss, val_loss = train_MANN_model(model=model, config=config, training_dataloader=train_loader, 
                                                       validation_dataloader=val_loader, PAE_model=PAE_model, tcnn=False, log_wandB=log_wandB, collect_phase=collect_phase)
            
            
            
        elif model_to_train == "MoETCNN":
            
            # Train
            train_loss, val_loss = train_MANN_model(model=model, config=config, training_dataloader=train_loader, 
                                                       validation_dataloader=val_loader, PAE_model=PAE_model, tcnn=True, log_wandB=log_wandB, collect_phase=collect_phase)
            
            print("Trained the model succesfully - plotting will probably fail")
            
        elif model_to_train == "LSTM":
            
            # Train
            train_loss, val_loss = train_predictor_model(model=model, config=config, training_dataloader=train_loader, 
                                                       validation_dataloader=val_loader, tcnn=True, log_wandB=log_wandB) 
            
        
        
        
        if save_file:
            # Save the Model
            saved_models_dir = os.path.join(BASE_DIR, "Saved Models")
            os.makedirs(saved_models_dir, exist_ok=True)
            model_save_location = os.path.join(saved_models_dir, datetime.now().strftime('%Y%m%d_%H%M') + "_" + config["training_tag"] + ".pth")
            torch.save(model.state_dict(), model_save_location)
            
            model_dict = {
                # Architecture
                "model_class": model.__class__.__name__,
                "pae_input_size": 21,
                "mann_input_size": 46,
                "output_size": 20,
                "pae_window_size": 201,
                "mann_window_size": 50,
                # "hidden_size": model.hidden_size,
                # "num_layers": model.num_layers,

                # Training hyperparameters
                "optimizer": "Adam",
                "learning_rate": config["lr"],
                "loss_fn": "MSELoss",

                # Data normalization
                "data_input_mean": train_ds.input_mean.tolist(),
                "data_input_std": train_ds.input_std.tolist(),
                "data_output_mean": train_ds.output_mean.tolist(),
                "data_output_std": train_ds.output_std.tolist()
                }


            model_save_location = os.path.join(saved_models_dir, datetime.now().strftime('%Y%m%d_%H%M') + "_" + config["training_tag"])
            with open(model_save_location + '.json', 'w') as file:
                json.dump(model_dict, file, indent=4)
        
        # Plot the training and validation loss curves        
        # utility.plot_train_val_loss(train_loss, val_loss)
        
    else:
        
        # Load a model
        weights = torch.load(model_file_path, weights_only=True)
        model.load_state_dict(weights)
    
    if model_to_train == "TCNN":
        if future_forcast:
            
            utility.stats_predictor_cal(train_loader_plot, model, df.columns[46:66])
            # utility.plot_predictor_results(train_loader_plot, model, df.columns[46:66], window=time_horizon_prediction)

            utility.stats_predictor_cal(val_loader_plot, model, df.columns[46:66])
            # utility.plot_predictor_results(val_loader_plot, model, df.columns[46:66], window=time_horizon_prediction)            
            
        else:            
            utility.plot_prediction(train_loader_plot, model, df.columns[46:66], tcnn=True)
            utility.plot_prediction(val_loader_plot, model, df.columns[46:66], tcnn=True)
            
        plt.show()
        
    elif model_to_train == "PAE":
        # Label columns must match the 21-channel PAE_input slice (df columns 21:42)
        # used by GroupedSequenceDataset - plot_PAE_recon assumes 21 channels.
        utility.plot_PAE_recon(train_loader_plot, model, config["seq_length"], df.columns[21:42])
        utility.plot_PAE_recon(val_loader_plot, model, config["seq_length"], df.columns[21:42])
        plt.show()
        
    elif model_to_train == "MANN":
        
        if future_forcast:
            utility.stats_predictor_cal(train_loader_plot, model, df.columns[46:66], PAE_model=PAE_model, mann=True)
            # utility.plot_predictor_results(train_loader_plot, model, df.columns[46:66], window=time_horizon_prediction, PAE_model=PAE_model, mann=True)

            utility.stats_predictor_cal(val_loader_plot, model, df.columns[46:66], PAE_model=PAE_model, mann=True)
            # utility.plot_predictor_results(val_loader_plot, model, df.columns[46:66], window=time_horizon_prediction, PAE_model=PAE_model, mann=True)        
        else:
            utility.plot_MANN_predictions(train_loader_plot, PAE_model, model, df.columns[32:44], tcnn=False)
            utility.plot_MANN_predictions(val_loader_plot, PAE_model, model, df.columns[32:44], tcnn=False)
        plt.show()
    
    elif model_to_train == "FCNN_SW":
        if future_forcast:
            
            utility.stats_predictor_cal(train_loader_plot, model, df.columns[46:66], fcnn_sw=True)
            # utility.plot_predictor_results(train_loader_plot, model, df.columns[46:66], window=time_horizon_prediction, fcnn_sw=True)

            utility.stats_predictor_cal(val_loader_plot, model, df.columns[46:66], fcnn_sw=True)
            # utility.plot_predictor_results(val_loader_plot, model, df.columns[46:66], window=time_horizon_prediction, fcnn_sw=True)            
            
        else:            
            utility.plot_prediction(train_loader_plot, model, df.columns[46:66], tcnn=False)
            utility.plot_prediction(val_loader_plot, model, df.columns[46:66], tcnn=False)
        plt.show()   

    elif model_to_train == "MoETCNN":
        
        if future_forcast:
            
            utility.stats_predictor_cal(train_loader_plot, model, df.columns[46:66], PAE_model=PAE_model, moe_tcnn=True)
            utility.plot_predictor_results(train_loader_plot, model, df.columns[46:66], window=time_horizon_prediction, PAE_model=PAE_model, moe_tcnn=True)

            utility.stats_predictor_cal(val_loader_plot, model, df.columns[46:66], PAE_model=PAE_model, moe_tcnn=True)
            utility.plot_predictor_results(val_loader_plot, model, df.columns[46:66], window=time_horizon_prediction, PAE_model=PAE_model, moe_tcnn=True) 
            
        else:
            utility.plot_MANN_predictions(train_loader_plot, PAE_model, model, df.columns[46:66], tcnn=True)
            utility.plot_MANN_predictions(val_loader_plot, PAE_model, model, df.columns[46:66], tcnn=True)
        plt.show()
    elif model_to_train == "LSTM":
        
        # pass
        if future_forcast:
            
            utility.stats_predictor_cal(train_loader_plot, model, df.columns[46:66])
            # utility.plot_predictor_results(train_loader_plot, model, df.columns[46:66], window=time_horizon_prediction)

            utility.stats_predictor_cal(val_loader_plot, model, df.columns[46:66])
            # utility.plot_predictor_results(val_loader_plot, model, df.columns[46:66], window=time_horizon_prediction)            
            
        else:            
            utility.plot_prediction(train_loader_plot, model, df.columns[46:66], tcnn=True)
            utility.plot_prediction(val_loader_plot, model, df.columns[46:66], tcnn=True)
        plt.show()   
        
    return 0
        

if __name__ == "__main__":
    raise SystemExit(main())