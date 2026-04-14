
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





def main():

    K = [1,5,20,50,80,100]
    
    # Joint Angle MAE
    MANN_angle_mae = [0.1094,0.1122,0.0998,0.1084,0.0937,0.0974]
    TCNN_angle_mae = [0.0978,0.0967, 0.0907,0.0952,0.1035,0.1016]
    FCNN_SW_angle_mae = [0.1282,0.1293,0.1326,0.1211,0.1383,0.1277]
    LSTM_angle_mae = [0.1190,0.1218,0.1238,0.1325,0.1255,0.1280]
    
    # Joint Angle RMSE
    MANN_angle_rmse = [0.1375,0.1365,0.1275,0.1306,0.1283,0.1329]
    TCNN_angle_rmse = [0.1690,0.1571,0.1365,0.1508,0.1550,0.1602]
    FCNN_SW_angle_rmse = [0.1767,0.1800,0.1818,0.1875,0.1960,0.2050]
    LSTM_angle_rmse = [0.1676,0.1732,0.1741,0.1893,0.1582,0.1755]
    
    # Joint Moment MAE
    MANN_moment_mae = [0.1561,0.1511,0.1406,0.1484,0.1291,0.1356]
    TCNN_moment_mae = [0.1423,0.1404,0.195,0.1460,0.1595,0.1569]
    FCNN_SW_moment_mae = [0.1783,0.1790,0.1818,0.1836,0.1838,0.1848]
    LSTM_moment_mae = [0.1642,0.1673,0.1833,0.1827,0.1685,0.1805]
        
    # Joint Moment RMSE
    MANN_moment_rmse = [0.2105,0.2008,0.1903,0.2057,0.1761,0.1848]
    TCNN_moment_rmse = [0.2025,0.1990,0.2070,0.2197,0.2250,0.2275]
    FCNN_SW_moment_rmse = [0.2525,0.2550,0.2650,0.2660,0.2750,0.2775]
    LSTM_moment_rmse = [0.2356,0.2450,0.2576,0.2601,0.2440,0.2552]
    
    plt.figure(figsize=(7, 3.5))
    plt.xlim(0,100)
    plt.ylim(0.1,0.275)
    # lines with dots on every point
    plt.plot(K, MANN_angle_rmse, marker="o", markersize=8, linewidth=2.5, color="blue", label="PAE-MoENN")
    plt.plot(K, TCNN_angle_rmse, marker="s", markersize=8, linewidth=2.5, color="red", alpha=0.75, label="TCNN")
    plt.plot(K, FCNN_SW_angle_rmse, marker="^", markersize=8, linewidth=2.5, color="green", alpha=0.75, label="FCNN-SW")
    plt.plot(K, LSTM_angle_rmse, marker="d", markersize=8, linewidth=2.5, color="orange", alpha=0.75, label="LSTM")

    plt.xlabel("Prediction Horizon", fontsize=12.0)
    plt.ylabel("Root Mean Square Error (rad)", fontsize=12.0)
    # plt.ylabel("Mean Absolute Error(rad)", fontsize=16.0)
    plt.grid(True, which="both", alpha=0.45)
    # plt.legend(fontsize=10.0)
    plt.tight_layout()
    plt.title("Joint Angles", fontsize=12.0)
    plt.show()
    
    # plt.figure(figsize=(7, 3.5))
    # plt.xlim(0,100)
    # plt.ylim(0,0.3)
    # # lines with dots on every point
    # plt.plot(K, MANN_angle_mae, marker="o", markersize=4, linewidth=2, color="blue", label="PAE-MoENN")
    # plt.plot(K, TCNN_angle_mae, marker="s", markersize=4, linewidth=2, color="red", label="TCNN")
    # plt.plot(K, FCNN_SW_angle_mae, marker="^", markersize=4, linewidth=2, color="green", label="FCNN-SW")
    # plt.plot(K, LSTM_angle_mae, marker="d", markersize=4, linewidth=2, color="orange", label="LSTM")

    # plt.xlabel("Prediction Time Step", fontsize=16.0)
    # # plt.ylabel("Root Mean Square Error (rad)", fontsize=16.0)
    # plt.ylabel("Mean Absolute Error(rad)", fontsize=16.0)
    # plt.grid(True, which="both", alpha=0.45)
    # plt.legend(fontsize=12.0)
    # plt.tight_layout()
    # plt.show()
    
    
    # plt.figure(figsize=(7, 3.5))
    # plt.xlim(0,100)
    # plt.ylim(0,0.25)
    # # lines with dots on every point
    # plt.plot(K, MANN_moment_mae, marker="o", markersize=4, linewidth=2, color="blue", label="PAE-MoENN")
    # plt.plot(K, TCNN_moment_mae, marker="s", markersize=4, linewidth=2, color="red", label="TCNN")
    # plt.plot(K, FCNN_SW_moment_mae, marker="^", markersize=4, linewidth=2, color="green", label="FCNN-SW")
    # plt.plot(K, LSTM_moment_mae, marker="d", markersize=4, linewidth=2, color="orange", label="LSTM")

    # plt.xlabel("Prediction Time Step", fontsize=16.0)
    # # plt.ylabel("Root Mean Square Error (rad)", fontsize=16.0)
    # plt.ylabel("Mean Absolute Error(Nm/kg)", fontsize=16.0)
    # plt.grid(True, which="both", alpha=0.45)
    # plt.legend(fontsize=12.0)
    # plt.tight_layout()
    # plt.show()
    
    plt.figure(figsize=(7, 3.5))
    plt.xlim(0,100)
    plt.ylim(0,0.30)
    # lines with dots on every point
    plt.plot(K, MANN_moment_rmse, marker="o", markersize=8, linewidth=2.5, color="blue", label="PAE-MoENN")
    plt.plot(K, TCNN_moment_rmse, marker="s", markersize=8, linewidth=2.5, color="red", alpha=0.75, label="TCNN")
    plt.plot(K, FCNN_SW_moment_rmse, marker="^", markersize=8, linewidth=2.5, color="green", alpha=0.75, label="FCNN-SW")
    plt.plot(K, LSTM_moment_rmse, marker="d", markersize=8, linewidth=2.5, color="orange", alpha=0.75, label="LSTM")

    plt.xlabel("Prediction Horizon", fontsize=12.0)
    plt.ylabel("Root Mean Square Error (Nm/kg)", fontsize=12.0)
    # plt.ylabel("Mean Absolute Error(rad)", fontsize=16.0)
    plt.grid(True, which="both", alpha=0.45)
    plt.legend(fontsize=12.0)
    plt.tight_layout()
    plt.title("Joint Moments", fontsize=12.0)
    plt.show()
    
    
    mae = [17.13518866430711, 16.742423452837443, 16.486075529527774, 16.30869345536059, 16.170078460903547, 16.07040239103971, 16.005157900076263, 15.992881048189055, 15.99596500059077, 16.01302153595624, 16.059833995612223, 16.13954994773398, 16.20249672561984, 16.277286239540405, 16.377524029833722, 16.489370167965188, 16.57492482655108, 16.681009970804947, 16.76910720721565, 16.885732192895805, 16.97308275377504]
    mae_v = [101.08104228511041, 98.43349111773556, 96.77481108421017, 95.76964164209579, 95.04744141780779, 94.55598187111305, 94.26389610984589, 94.01860469836957, 94.01994514245298, 94.03183103319279, 94.20382785678454, 94.46060502281708, 94.70760786399558, 94.99086600978178, 95.29488820471325, 95.62248687493123, 95.82138327348936, 95.84449129355083, 96.09960861703013, 96.0867503231706, 96.15571154926555]
    r2 =[-2.037788959697954, -1.887485024222777, -1.798628639483931, -1.7470024882877322, -1.7179523512851969, -1.6987220295677337, -1.6862688627324958, -1.6841391097896996, -1.693467790796975, -1.6971199959943952, -1.7128000441720683, -1.7370585373517669, -1.7477069021017142, -1.7745829513733455, -1.7918665623153789, -1.81840637176321, -1.8292367911344427, -1.8242234682553375, -1.839811609107012, -1.8346536441427927, -1.8358649080695715]
    
    mae =[6.5244593105263995, 6.468571002570105, 6.476332623626982, 6.5060417397587145, 6.577448532605827, 6.672547809179294, 6.783670847441676, 6.9041420633884645, 7.0390659835099685, 7.186971411379734, 7.336561303749625, 7.483500823565717, 7.629408820687431, 7.777827302935899, 7.928980961606604, 8.082398549893826, 8.242446751224094, 8.406913092638796, 8.571584016125687, 8.731632735245887, 8.858784131699364]
    mae_v = [26.50990770156119, 24.65037989773562, 25.912412591231398, 28.430856937094315, 30.782012050114204, 32.730719221838875, 34.37089023940673, 35.79980153699253, 37.169747837032034, 38.60400740350886, 40.0162855605397, 41.27788624454493, 42.3390892347066, 43.229451362162465, 44.00362342154581, 44.64950075160222, 45.22895607000667, 45.77309884217633, 46.25332440726678, 46.7017182686709, 47.0977345405603]
    r2 = [0.7873366471264686, 0.8243884350438517, 0.8058826593319798, 0.7613389460626458, 0.7120780245958416, 0.6657810916266267, 0.6229786379037345, 0.5830777267242686, 0.5437166929792567, 0.5028748298343975, 0.46221788778458006, 0.4249927592343816, 0.3918680297727227, 0.36141037529008, 0.33350401270098695, 0.3097151289737843, 0.28958037035580897, 0.27255560416947544, 0.25755054288420787, 0.2423766381431911, 0.2290321606110567]


    print(sum(mae)/len(mae))
    print(sum(mae_v)/len(mae_v))
    print(sum(r2)/len(r2))
    
    
    
    
#     # Load data
# # data_path = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data - Second Skin/Walking_bio_torque_new.csv"
# data_path_2 = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data - Second Skin/squatting_bio_torque_new.csv"
# df = pd.read_csv(data_path_2)
    
# # ---- SETTINGS ----
# start_idx = 13560
# end_idx   = 19000
# fs = 1000  # 👈 CHANGE THIS to your sampling frequency (Hz)

# selected_cols = [
#     " root.main.LinkHardwareController.LinkControllerAndEstimator.AugmentativeControllerManager.BioTorqueController.desiredAppliedBioTorque_l_hip_x",
#     " root.main.LinkHardwareController.LinkControllerAndEstimator.AugmentativeControllerManager.BioTorqueController.desiredAppliedBioTorque_l_hip_y",
#     " root.main.LinkHardwareController.LinkControllerAndEstimator.AugmentativeControllerManager.BioTorqueController.desiredAppliedBioTorque_l_knee_y",
# ]

# # Custom labels (edit as you like)
# custom_labels = [
#     "Hip Abduction/Adduction Torque (Nm)",
#     "Hip Flexion/Extension Torque (Nm)",
#     "Knee Torque (Nm)"
# ]

# # ---- Extract slice ----
# time_samples = np.arange(end_idx - start_idx)
# time_seconds = time_samples / fs  # Convert to seconds

# fig, axes = plt.subplots(len(selected_cols), 1, figsize=(4, 6), sharex=True)

# if len(selected_cols) == 1:
#     axes = [axes]

# for ax, col, label in zip(axes, selected_cols, custom_labels):
    
#     data_slice = df[col].iloc[start_idx:end_idx].values
    
#     ax.set_xlim(0, 5)
#     # 🔥 Align to zero (subtract first value)
#     data_slice = data_slice - data_slice[0]
    
#     ax.plot(time_seconds, data_slice, color="blue", linewidth=2.0)
    
#     ax.set_title(label, fontsize=10)
#     # ax.set_ylabel(label)
#     ax.grid(True)

# axes[-1].set_xlabel("Time (seconds)", fontsize=10)
# # fig.suptitle("Real Time Torque Prediction", fontsize=14)

# plt.tight_layout()
# plt.show()


if __name__ == "__main__":
    raise SystemExit(main())