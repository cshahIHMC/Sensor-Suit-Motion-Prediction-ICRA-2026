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
from itertools import islice
from Library import utility

    
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

# Function to calculate the PAE validation loss
def cal_validation_loss(model, validation_dataloader, lossFn, lossFn_no_reduction):
    model.eval()
    
    val_loss = 0.0
    individual_losses = np.zeros(21, dtype=np.float32)
    
    # Convoluted Signal
    all_latents = []
    
    # Reconstructed signal
    all_signals = []
    
    with torch.no_grad():
        for batch in validation_dataloader:
            
            PAE_inputs, _, _ = batch
            PAE_inputs = utility.ToDevice(PAE_inputs)
                        
            # Predict
            # outputs, latent, signal, params  = model(PAE_inputs)
            outputs, _, _, _  = model(PAE_inputs)
                    
            # Flattening the outputs and inputs and calculating the loss
            flattened_inputs = PAE_inputs.reshape(PAE_inputs.shape[0], -1)
            flattened_outputs = outputs.reshape(outputs.shape[0], -1)
        
            # Calculate Total Loss
            loss = lossFn(flattened_inputs, flattened_outputs)
            
            # Calculate running loss
            val_loss += loss.item() * PAE_inputs.size(0)
            
            # Calculate Individual Loss, first across sequence length and then across batches
            individual_loss = lossFn_no_reduction(PAE_inputs, outputs)
            individual_loss_across_sequence_length_across_batch = individual_loss.mean(2).mean(0)
            
            individual_losses = individual_losses + ( utility.Item(individual_loss_across_sequence_length_across_batch).numpy() * PAE_inputs.size(0))
            
             # ensure no lingering references
            del outputs, individual_loss_across_sequence_length_across_batch, loss
            
            # Append all the latents and signals
            # all_latents.append(utility.Item(latent))
            # all_signals.append(utility.Item(signal))  
            
        
        val_loss = val_loss / len(validation_dataloader.dataset)
        individual_losses = individual_losses / len(validation_dataloader.dataset)   
        
        
    return val_loss, individual_losses

## Training Function
def train_model(model, config, training_dataloader, validation_dataloader, log_wandB=False):
    
    ## Setting up an optimizer and a loss function - Original Paper used a AdamWr optimizer We using a simple SGD
    learning_rate = config["lr"]
    momentum = config["momentum"]
    
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
        print(len(training_dataloader))
        print(batch[0].shape)
        break
    
    ## Training Loop
    for epoch in range(epochs):
    
        running_loss = 0.0
    
        for batch in training_dataloader:
        
            PAE_inputs, _, _ = batch
            PAE_inputs = utility.ToDevice(PAE_inputs)

            # Zero the parameter gradients
            optimizer.zero_grad()
        
            # Forward
            outputs, latent, signal, params  = model(PAE_inputs)
        
            # Flattening the outputs and inputs and calculating the loss
            flattened_inputs = PAE_inputs.reshape(PAE_inputs.shape[0], -1)
            flattened_outputs = outputs.reshape(outputs.shape[0], -1)
        
            # Calculate Loss
            loss = lossFn(flattened_inputs, flattened_outputs)
        
            # Backward
            loss.backward()
            optimizer.step()
        
            # Calculate running loss
            running_loss += loss.item() * PAE_inputs.size(0)
                                             
        train_loss = running_loss / len(training_dataloader.dataset)
        training_losses.append(train_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss}')
    
    
        val_loss, individual_loss = cal_validation_loss(model, validation_dataloader, lossFn, lossFn_no_reduction)
        validation_losses.append(val_loss)
        individual_losses.append(individual_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
        
        
        if log_wandB:
            wandb.log({"train/train_loss": train_loss,
                        "train/epoch": epoch,
                        "val/val_loss": val_loss,
                        "val/epoch":epoch})
            
    print('Finished Training')
    
    return training_losses, validation_losses


def plot_df(dataloader, model, file_name="", col_names=""):
    
    model.eval()
    fig, axs = plt.subplots(3, 7, figsize=(30,10), sharey=True, sharex=True)
    
    step = 50
    end_plot_timestep = 20000
    with torch.no_grad():
        
        for j, batch in enumerate(islice(dataloader, 0, end_plot_timestep, step)):
            
            start_index = j*step
            
            end_index = start_index + 201
            
            # Transpose and convert to tensor
            input_tensor, _, _ = batch

            output,_,_,_ = model(input_tensor)
        
            output = output.squeeze(0)
            input = input_tensor.squeeze(0)
    
            output_df = pd.DataFrame(output.T.numpy())
            input_df = pd.DataFrame(input.T.numpy())
            
            for i in range(21):
                row = i % 3
                col = i // 3
                ax = axs[row, col]
            
                
                
                
                if end_index>=end_plot_timestep: 
                    
                    value_over = end_index - end_plot_timestep
                    stop_plot = len(output_df) - value_over
                    ax.plot(dataloader.dataset.indices[start_index:end_plot_timestep],output_df.iloc[:stop_plot,i], linewidth=1, alpha=0.75)
                    
                    # Plot the ground truth
                    # TODO - Right now it plots lines on top of each other - Only plot once
                    ax.plot(dataloader.dataset.indices[start_index:end_plot_timestep], input_df.iloc[:stop_plot, i], linewidth=1, color="black")  # Plot the i-th column
                
                else:
                    ax.plot(dataloader.dataset.indices[start_index:end_index],output_df.iloc[:,i], linewidth=1, alpha=0.75)
                    
                     # Plot the ground truth
                     # TODO - Right now it plots lines on top of each other - Only plot once
                    ax.plot(dataloader.dataset.indices[start_index:end_index], input_df.iloc[:end_index, i], linewidth=1, color="black")  # Plot the i-th column
                
                
                # ax.set_ylim(-10,8)
                
                # Name of the Sub Plot 
                
                # if j==0:
                #     joint_name = None
                #     prefix = col_names[i][:4]
                    
                #     for k in key_list:
                        
                #         if prefix in k:
                #             joint_name = imu_joint_map[k]
                #             break

                #     if "_l" in joint_name:
                #         joint_name = joint_name.replace("_l", "")
                #         joint_name = "left " + joint_name
                #     elif "_r" in joint_name:
                #         joint_name = joint_name.replace("_r", "")
                #         joint_name = "right " + joint_name

                    
                    # name = joint_name + " (" + col_names[i] + ")"
                # name = col_names[i]
                # ax.set_title(name)
                ax.tick_params(labelsize=8)
            
            
    # fig.suptitle(file_name)

    plt.tight_layout()
    plt.show()
    # plt.savefig(file_name, dpi=300, bbox_inches='tight')
    

def main():
    
    # Different Flags
    # Logging False
    log_wandB = True
    train_and_plot = True
    
    file_name = "PAE training Scherpeel Dataset - 10 Subjects 10 Phases - 40 epochs"
    project_name = "ICRA 2026"
    
    # COnfig the configurations
    config = {
        "training_tag": file_name,
        "project_name": project_name,
        "epochs": 40,
        "batch_size": 128,
        "num_workers": 8,
        "momentum":0.9,
        "lr": 1e-4,
        "dropout": 0.0,
        "dataset": "IHMC Senorsuit",
        "seq_length": 201,
        "pred_length": 1,
        "inputs": 21,
        "outputs": 21,
        "phases": 10,
        "intermediate_channels": 16,
        "training_window": 1.0, # How many seconds of data you are reviewing
        "data_recorded_rate": 200 # 
    }
    
    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        project_name = config["project_name"]
        wandb.init( project=project_name, name= config["training_tag"], config=config)
    
    # Data setup
    data_path = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data/all_subjects_req_sim_data.csv"
    df = pd.read_csv(data_path)
    
    print("Full df Shape: ", df.shape)
    
    cols_2_get = col_2_extract()
    df_pae = df[cols_2_get]
    
    print("Extracted df Shape: ", df_pae.shape)
    
    subjects, conditions, all_pairs = utility.extract_pairs(df_pae, "subject", "condition")
    groups = utility.make_group_dict(df, "subject", "condition", "time")
    
    # 4) Split by pairs
    train_pairs,val_pairs = utility.split_pairs_train_val(all_pairs, val_frac=0.15)
    
    train_ds = GroupedSequenceDataset(
        df_pae, seq_len=config["seq_length"], pred_len=config["pred_length"], stride=1,
        time_col="time", subject_col="subject", condition_col="condition",
        include_groups=train_pairs
        )
    
    print("Training Samples: ", len(train_ds))
    
    val_ds = GroupedSequenceDataset(
        df_pae, seq_len=config["seq_length"], pred_len=config["pred_length"], stride=1,
        time_col="time", subject_col="subject", condition_col="condition",
        include_groups=val_pairs
        )
    print("Validation Samples: ", len(val_ds))
    
    train_sampler = GroupedBatchSampler(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=False)
    train_loader  = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    
    val_sampler = GroupedBatchSampler(val_ds, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    val_loader  = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    
    train_sampler_plot = GroupedBatchSampler(train_ds, batch_size=1, shuffle=False, drop_last=False)
    train_loader_plot  = DataLoader(train_ds, batch_sampler=train_sampler_plot, num_workers=8, pin_memory=True)
    
    val_sampler_plot = GroupedBatchSampler(val_ds, batch_size=1, shuffle=False, drop_last=False)
    val_loader_plot  = DataLoader(val_ds, batch_sampler=val_sampler_plot, num_workers=8, pin_memory=True)
    
    
    
    # all_feat0 = [] 
    # i = 0
    # for batch, meta in train_loader:
    #     x_bt = batch[:, 1, :]   # take feature 0 → [B, T]
    #     # print(x_bt.shape)
    #     all_feat0.append(x_bt.detach().cpu().numpy().reshape(-1))  # → [B*T]
    #     # print(len(all_feat0))
    #     # break
        
    #     i = i+1
        
    #     if i%1000 == 0:
    #         print(i)
    #     if i == 20000:
    #         break
    
    # feat0 = np.concatenate(all_feat0, axis=0)   # 1D: all batches/time concatenated
    # print(len(feat0))
    
    # plt.figure()
    # plt.plot(feat0)
    # plt.title("First feature across all batches (concatenated)")
    # plt.xlabel("Concatenated time index (B×T)")
    # plt.ylabel("Feature 0 value")
    # plt.show()
    
    
    # Model Setup
    
    if train_and_plot:
        model = utility.ToDevice(PAE.Model(
                              input_channels=config["inputs"],
                              embedding_channels=config["phases"],
                              intermediate_channels=config["intermediate_channels"],
                              time_range=config["seq_length"],
                              window=config["training_window"]
                             ))

        # # Train Model
        training_losses, validation_losses = train_model(model=model, config=config, training_dataloader=train_loader, 
                                                       validation_dataloader=val_loader, log_wandB=log_wandB)

        # Save the Model
        model_save_location = "Saved Models/" + datetime.now().strftime('%Y%m%d_%H%M') + "_" + config["training_tag"] + ".pth"
        torch.save(model.state_dict(), model_save_location)

        model = model.to("cpu")
        
    else:
        
        model_location = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Saved Models/20250829_0213_PAE training Scherpeel Dataset - 10 Subjects 10 Phases - 25 epochs.pth"
        weights = torch.load(model_location, weights_only=True)
        model = PAE.Model(
                          input_channels=config["inputs"],
                          embedding_channels=config["phases"],
                          intermediate_channels=config["intermediate_channels"],
                          time_range=config["seq_length"],
                          window=config["training_window"]
                         )
        model.load_state_dict(weights)
    
    
    # # Plot all the different plots
    # plot_df(train_loader_plot, model)
    # plot_df(val_loader_plot, model)
    


if __name__ == "__main__":
    raise SystemExit(main())
    