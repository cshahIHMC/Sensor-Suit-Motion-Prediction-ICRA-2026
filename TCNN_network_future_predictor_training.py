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
from Models.TCNN import TCNModel, TCNModel_Forecast
from Models.TCNN_MOE import MANN_TCN_DynamicWeights, MANN_TCN_DynamicWeights_Forecast, MANN_TCN_DynamicWeights_Forecast_k
from Models import PAE
from itertools import islice
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

# Function to calculate the PAE validation loss
def calc_val_loss(model, PAE_model, validation_dataloader, lossFn, lossFn_no_reduction=None):
    model.eval()
    PAE_model.eval()
    
    val_loss = 0.0
    # individual_losses = np.zeros(6, dtype=np.float32)
    
    with torch.no_grad():
        for batch in validation_dataloader:
            
            PAE_inputs, FCNN_inputs, FCNN_outputs = batch
            PAE_inputs = utility.ToDevice(PAE_inputs)
                        
            # # Predict
            _, _, _, params  = PAE_model(PAE_inputs)
            
            # flattened_inputs = utility.ToDevice(FCNN_inputs.reshape(FCNN_inputs.shape[0], -1))

            params_cat = torch.cat(params, dim=2)
            phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
            phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
            phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])
            
            phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 
            phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)
            
            # TCNN 1 time step prediction
            # y_pred = model(utility.ToDevice(FCNN_inputs))
            
            # TCNN future forecasting
            # y_pred = model(utility.ToDevice(FCNN_inputs))
            
            FCNN_inputs = utility.ToDevice(FCNN_inputs)
            B = FCNN_inputs.size(0)
            H = model.horizon
            
            k = torch.randint(1, H+1, (1,), device=FCNN_inputs.device) 
            
            # TCNN_MOE
            y_pred = model(phaseInputs, FCNN_inputs, k)
            
            FCNN_outputs = utility.ToDevice(FCNN_outputs[:,:,0:k])
            
            # Calculate the loss
            loss = lossFn(y_pred,FCNN_outputs)

            
            # Calculate running loss
            val_loss += loss.item() * PAE_inputs.size(0)
            
            # # Calculate Individual Loss, first across sequence length and then across batches
            # individual_loss = lossFn_no_reduction(ToDevice(FCNN_outputs), y_pred)
            # individual_loss_across_batch = individual_loss.mean(0)

            # individual_losses = individual_losses + ( Item(individual_loss_across_batch).numpy() * FCNN_inputs.size(0))

        val_loss = val_loss / len(validation_dataloader.dataset)
        # individual_losses = individual_losses / len(validation_dataloader.dataset)   

    return val_loss

## Training Function
def train_model(model, config, training_dataloader, validation_dataloader, PAE_model, log_wandB=False):   
    
    ## Setting up an optimizer and a loss function - Original Paper used a AdamWr optimizer We using a simple SGD
    learning_rate = config["lr"]
    momentum = config["momentum"]

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss function
    lossFn = nn.MSELoss()
    # lossFn_no_reduction = nn.MSELoss(reduction='none')
    
    for batch in training_dataloader:

        PAE_input, MANN_input, MANN_output = batch
        print("PAE_input Shape: ", PAE_input.shape)
        print("Inputs Shape: ", MANN_input.shape)
        print("Outputs Shape: ", MANN_output.shape)
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
        
            PAE_inputs, FCNN_inputs, FCNN_outputs = batch
            
            PAE_inputs = utility.ToDevice(PAE_inputs)
            
            PAE_model.eval()
            _, _, _, params  = PAE_model(PAE_inputs)
            
            params_cat = torch.cat(params, dim=2)
            phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
            phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
            phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])
            phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 
            phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)
            
            
            # TCNN Prediction
            # y_pred = model(utility.ToDevice(FCNN_inputs))
            
            # TCNN Forecasting
            # y_pred = model(utility.ToDevice(FCNN_inputs))
            
            FCNN_inputs = utility.ToDevice(FCNN_inputs)
            B = FCNN_inputs.size(0)
            H = model.horizon
            
            k = torch.randint(1, H+1, (1,), device=FCNN_inputs.device) 
            
            
            
            # TCNN_MOE
            y_pred = model(phaseInputs, FCNN_inputs, k)
            
            # Target 
            FCNN_outputs = utility.ToDevice(FCNN_outputs[:,:,0:k])
             
            # Calculate the loss
            loss = lossFn(y_pred, FCNN_outputs)

            # # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item() * FCNN_inputs.size(0)


                
        train_loss = running_loss / len(training_dataloader.dataset)
        training_losses.append(train_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss}')
        
        val_loss = calc_val_loss(model, PAE_model, validation_dataloader, lossFn)
        validation_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
        
        if log_wandB:
            wandb.log({"train/train_loss": train_loss,
                        "train/epoch": epoch,
                        "val/val_loss": val_loss,
                        "val/epoch":epoch})
        
            
    return training_losses, validation_losses    


def results(dataloader, PAE_model, model, col_names, plot_save_name=None):
    
    stats_calc(dataloader, PAE_model, model, col_names)
    # plot_results(dataloader, PAE_model, model, col_names)


def plot_results(dataloader, PAE_model, model, col_names, plot_save_name=None):
     
    model.eval()
    fig1, axs1 = plt.subplots(2, 5, figsize=(30,10), sharey=True, sharex=True)
    fig2, axs2 = plt.subplots(2, 5, figsize=(30,10), sharey=True, sharex=True)
    
    axs1_flat = axs1.flatten()
    axs2_flat = axs2.flatten()
    
    step = 5
    end_plot_timestep = 100
    
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
               
            PAE_inputs, FCNN_inputs, FCNN_outputs = batch
            PAE_inputs = utility.ToDevice(PAE_inputs)
            
            PAE_model.eval()
            _, _, _, params  = PAE_model(PAE_inputs)
            
            params_cat = torch.cat(params, dim=2)
            phaseInputs = params_cat.reshape(params_cat.shape[0], -1)
            phase_sin_x = torch.sin(2 * np.pi * params_cat[...,0])
            phase_cos_x = torch.cos(2 * np.pi * params_cat[...,0])
            phaseInputs = torch.stack([phase_sin_x, phase_cos_x, params_cat[...,1], params_cat[...,2], params_cat[...,3]], dim=2) 
            phaseInputs = phaseInputs.reshape(phaseInputs.shape[0], -1)
            
            # TCNN_MOE
            y_pred = model(phaseInputs, utility.ToDevice(FCNN_inputs),50)
            
            pred_np = utility.Item(y_pred).numpy()
            ground_truth_np = FCNN_outputs.numpy()
            
            batch_len = pred_np.shape[0]
            feature_len = pred_np.shape[1]
            
            for i in range(0, batch_len, step):
                
                start_index = j*batch_len + i
                end_index = start_index + dataloader.dataset.pred_len
                
                for k in range(10):
                    
                    axs1_flat[k].plot(dataloader.dataset.indices[start_index:end_index], pred_np[i,k,:], linewidth=1, alpha=1.0, color="red", label="Pred")
                    axs1_flat[k].plot(dataloader.dataset.indices[start_index:end_index], ground_truth_np[i,k,:], linewidth=1, alpha=0.75, color="black", label="Ground Truth")
                    axs1_flat[k].set_title(col_names[k])
                    
                    axs2_flat[k].plot(dataloader.dataset.indices[start_index:end_index], pred_np[i,k+10,:], linewidth=1, alpha=1.0, color="red", label="Pred")
                    axs2_flat[k].plot(dataloader.dataset.indices[start_index:end_index], ground_truth_np[i,k+10,:], linewidth=1, alpha=0.75, color="black", label="Ground Truth")
                    axs2_flat[k].set_title(col_names[k+10])
                    
            if j == end_plot_timestep:
                break
                         
    plt.tight_layout()
    plt.show()            
                
def stats_calc(dataloader, PAE_model, model, col_names,
                 plot_save_name=None):
    model.eval()
    PAE_model.eval()
  
    # ground_truth = []
    # preds = []
    
    device = next(model.parameters()).device
    out_std = torch.as_tensor(dataloader.dataset.output_std, device=device, dtype=torch.float64)  # [F]

    # Running accumulators (float64 for stability)
    sum_abs   = torch.zeros_like(out_std, dtype=torch.float64)  # Σ |e|
    sum_abs2  = torch.zeros_like(out_std, dtype=torch.float64)  # Σ |e|^2  (for std of abs error)
    sum_sq    = torch.zeros_like(out_std, dtype=torch.float64)  # Σ e^2    (for RMSE)
    count     = torch.tensor(0, dtype=torch.int64, device=device)  # total samples per feature


    with torch.no_grad():
        for batch in dataloader:
            
            PAE_inputs, FCNN_inputs, FCNN_outputs = batch
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
            
            
            # TCNN 1 Step prediction
            # y_pred = model(utility.ToDevice(FCNN_inputs))
            
            # TCNN future forecastor
            # y_pred = model(utility.ToDevice(FCNN_inputs))
            
            # TCNN MOE
            y_pred = model(phaseInputs, utility.ToDevice(FCNN_inputs), 50)

            # errors in ORIGINAL scale: (pred - gt) * std
            # broadcast std over [B, F, T]
            err = (y_pred - utility.ToDevice(FCNN_outputs)).to(dtype=torch.float64) * out_std.view(1, -1, 1)

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

def main():
    
    # Different Flags
    # Logging False
    log_wandB = True
    train_and_plot = True
    
    file_name = "Predictor training Scherpeel Dataset - 10 Subject - random forecast - MANN_TCN_DynamicWeights_Forecast_k(43,20,50,8,[32, 64, 64, 128, 32],50,256,3,0.2,0.2)"
    project_name = "ICRA 2026"
    
    # Config the configurations
    config = {
        "training_tag": file_name,
        "project_name": project_name,
        "epochs": 40,
        "batch_size": 128,
        "num_workers": 8,
        "momentum":0.9,
        "lr": 1e-4,
        "dataset": "IHMC Senorsuit",
        "seq_length": 201,
        "pred_length":100,
        "inputs": 43,
        "outputs": 20,
        "hidden_layers": 5,
        "hidden_neurons": 256,
        "dropout": 0.4
    }

    
    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        project_name = config["project_name"]
        wandb.init( project=project_name, name= config["training_tag"], config=config)
    
    # Data setup
    data_path = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data/all_subjects_req_sim_data.csv"
    PAE_model_file = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Saved Models/20250904_1754_PAE training Scherpeel Dataset - 10 Subjects 10 Phases - 40 epochs.pth"
    df = pd.read_csv(data_path)
    
    print("Full df Shape: ", df.shape)
    
    cols_2_get = col_2_extract()
    df_mann = df[cols_2_get]
    
    print("Extracted df Shape: ", df_mann.shape)

    subjects, conditions, all_pairs = utility.extract_pairs(df_mann, "subject", "condition")
    groups = utility.make_group_dict(df, "subject", "condition", "time")
    
    # 4) Split by pairs
    train_pairs,val_pairs = utility.split_pairs_train_val(all_pairs, val_frac=0.15)
    
    train_ds = GroupedSequenceDataset(
        df_mann, seq_len=config["seq_length"], pred_len=config["pred_length"], stride=1,
        time_col="time", subject_col="subject", condition_col="condition",
        include_groups=train_pairs
        )
    
    print("Training Samples: ", len(train_ds))
    
    val_ds = GroupedSequenceDataset(
        df_mann, seq_len=config["seq_length"], pred_len=config["pred_length"], stride=1,
        time_col="time", subject_col="subject", condition_col="condition",
        include_groups=val_pairs
        )
    print("Validation Samples: ", len(val_ds))
    
    train_sampler = GroupedBatchSampler(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=False)
    train_loader  = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
    
    val_sampler = GroupedBatchSampler(val_ds, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    val_loader  = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    
    train_sampler_plot = GroupedBatchSampler(train_ds, batch_size=64, shuffle=False, drop_last=False)
    train_loader_plot  = DataLoader(train_ds, batch_sampler=train_sampler_plot, num_workers=8, pin_memory=True)
    
    val_sampler_plot = GroupedBatchSampler(val_ds, batch_size=64, shuffle=False, drop_last=False)
    val_loader_plot  = DataLoader(val_ds, batch_sampler=val_sampler_plot, num_workers=8, pin_memory=True)
    

    
    # model = utility.ToDevice(Model(50,256,4,inputs,512,28, 0.3))
    
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
    
    # FCNN_inputs = config["FCNN_inputs"] * config["FCNN_seq_length"] + 4 * config["PAE_phases"]
    inputs = config["inputs"] * config["seq_length"]
    
    
    if train_and_plot:
        
        ## Load Model
        # Regular TCNN
        # model = utility.ToDevice(TCNModel(43,20,[64, 128, 128, 256, 64],2,0.2))
        
        # Forecasting TCNN
        # model = utility.ToDevice(TCNModel_Forecast(43,20,20,[64, 128, 128, 256, 64],2,0.2))
        
        # MoE style TCNN
        # model =utility.ToDevice(MANN_TCN_DynamicWeights(43,20,10,[64, 128, 128, 256, 64],50,256,2,0.2,0.2))
        # model = utility.ToDevice(MANN_TCN_DynamicWeights_Forecast(43,20,20,10,[64, 128, 128, 256, 64],50,256,2,0.2,0.2))
        model = utility.ToDevice(MANN_TCN_DynamicWeights_Forecast_k(43,20,100,8,[32, 64, 64, 128, 32],50,256,3,0.2,0.2))

        # Train
        training_losses, validation_losses = train_model(model=model, config=config, training_dataloader=train_loader, 
                                                       validation_dataloader=val_loader, PAE_model=PAE_model, log_wandB=log_wandB)


        # Save the Model
        model_save_location = "Saved Models/" + datetime.now().strftime('%Y%m%d_%H%M') + "_" + config["training_tag"] + ".pth"
        torch.save(model.state_dict(), model_save_location)

        # model = model.to("cpu")
    
    else:
        
        model_location = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Saved Models/20250904_1822_Predictor training Scherpeel Dataset - 10 Subject - random forecast - MANN_TCN_DynamicWeights_Forecast_k(43,20,50,8,[32, 64, 64, 128, 32],50,256,3,0.2,0.2).pth"
        weights = torch.load(model_location, weights_only=True)
        
        # model = utility.ToDevice(TCNModel(43,20,[64, 128, 128, 128, 256],2,0.2))
        
        # Forecasting TCNN
        # model = utility.ToDevice(TCNModel_Forecast(43,20,20,[64, 128, 128, 256, 64],2,0.2))
        
        # MoE style TCNN
        # model =utility.ToDevice(MANN_TCN_DynamicWeights(43,20,10,[64, 128, 128, 256, 64],50,256,2,0.2,0.2))
        # model = utility.ToDevice(MANN_TCN_DynamicWeights_Forecast(43,20,20,10,[64, 128, 128, 256, 64],50,256,2,0.2,0.2))
        
        model = utility.ToDevice(MANN_TCN_DynamicWeights_Forecast_k(43,20,100,8,[32, 64, 64, 128, 32],50,256,3,0.2,0.2))
        
        model.load_state_dict(weights)


    # results(train_loader_plot, PAE_model, model, cols_2_get[43:63])
    results(val_loader_plot, PAE_model, model, cols_2_get[43:63])



if __name__ == "__main__":
    raise SystemExit(main())