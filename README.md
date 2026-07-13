# Sensor-Suit Motion Prediction (IROS 2026)

**Author:** Chinmay Shah
**Institution:** Institute for Human and Machine Cognition (IHMC) / University of West Florida (UWF)

This repository contains the training, evaluation, and plotting code used for our
IROS 2026 submission on predicting future lower-body joint angles and joint
moments from a wearable IMU sensor suit, at prediction horizons ranging from a
single timestep out to 100 timesteps ahead.

## Method overview

The pipeline has two stages:

1. **Periodic Autoencoder (PAE)** (`Models/PAE.py`) — a 1D-convolutional
   autoencoder that learns per-channel phase, frequency, amplitude, and offset
   parameters describing the periodicity of each IMU signal, and reconstructs the
   input from a sinusoidal latent representation built from those parameters. This
   architecture is adapted from the Periodic Autoencoder introduced in DeepPhase
   (Starke et al., SIGGRAPH 2022) — see [Citations](#citations).
2. **Motion predictor** — the PAE's phase parameters gate a downstream predictor
   that forecasts future joint angles/moments. Several predictor architectures are
   implemented so they can be compared against each other:
   - `Models/MANN.py` — Mode-Adaptive Neural Network: a gating network blends a
     mixture of "expert" linear layers using the PAE phase features.
   - `Models/TCNN_MOE.py` — a dynamic-weight Temporal Convolutional Network
     mixture-of-experts (the primary "PAE-MoENN" model used in the paper); the
     gate blends TCN convolution kernels directly rather than only blending
     outputs. `Models/TCNN_MOE_Additional.py` contains earlier exploratory MoE
     variants that are not part of the main pipeline.
   - `Models/TCNN.py` — plain TCN baseline (no phase gating).
   - `Models/FCNN.py` — fully-connected sliding-window baseline ("FCNN-SW").
   - `Models/LSTM.py` — LSTM baseline with direct multi-horizon prediction.

## Repository structure

```
.
├── data_extraction.py        # Converts the raw per-subject dataset into a single training CSV
├── network_training.py       # Main entry point: builds, trains, saves, and evaluates a model
├── DataLoader/
│   └── data_loader_pae.py    # PyTorch Dataset/Sampler: builds (subject, condition)-grouped windows
├── Library/
│   └── utility.py            # Shared helpers: device transfer, splitting, plotting, statistics
├── Models/
│   ├── PAE.py                # Periodic Autoencoder (phase/frequency/amplitude/offset extraction)
│   ├── MANN.py                # Mode-Adaptive gated mixture-of-experts predictor
│   ├── TCNN_MOE.py           # Dynamic-weight TCN mixture-of-experts predictor (main MoE model)
│   ├── TCNN_MOE_Additional.py# Exploratory MoE-TCN variants (not used by the main pipeline)
│   ├── TCNN.py                # Plain TCN baseline
│   ├── FCNN.py                # Fully-connected sliding-window baseline
│   ├── LSTM.py                 # LSTM baseline
│   └── Autoencoder.py         # Generic FCNN-style autoencoder (general utility, not part of the pipeline)
├── joint_angle_plot.py       # Generates joint-angle prediction-vs-ground-truth figures
├── plot_error_plots.py       # Computes per-joint MAE/STD/RMSE at a fixed horizon k
├── plot_results.py           # Plots RMSE-vs-horizon curves from the paper's result tables
├── stats_cal.py              # Evaluates a saved checkpoint on held-out test-subject data
├── test_functions.py         # Scratch/dev utilities used while building the DataLoader
└── Data/, Data - Second Skin/, Saved Models/, wandb/, Plots/   # Git-ignored (see Dataset Setup)
```

## Dataset setup

Training data is **not** included in this repository (the `Data/`, `Data - Second
Skin/`, and `Saved Models/` folders are git-ignored — see `.gitignore`). To
reproduce the pipeline:

1. **Download the dataset.** The wearable sensor-suit dataset used here ("Second
   Skin") is described in:

   > Casey, R. T. F., Nuesslein, C. P. O., Davenport, F., Wheeler, J., Mazumdar,
   > A., Sawicki, G., & Young, A. J. *The Second Skin: A Wearable Sensor Suite
   > That Enables Real-Time Human Biomechanics Tracking Through Deep Learning.*
   > IEEE Transactions on Biomedical Engineering, 73(2), 621–630, 2026.
   > https://pubmed.ncbi.nlm.nih.gov/40668712/

   Follow the link above for the paper and its data-availability statement to
   obtain the open-source release of the dataset (IMU, insole pressure, EMG, and
   ground-truth joint kinematics/kinetics for 10 subjects across 33
   construction/hazardous-cleanup tasks).

2. **Place the data** under a local `Data - Second Skin/OpenSource_Dataset/Data/`
   folder (matching the per-subject/per-task layout the dataset ships with, e.g.
   `Data/AB01/IMU/<task_name>/...`).

3. **Build the combined training CSV.** `data_extraction.py` walks each subject's
   task folders, downsamples the signals, tags rows with subject/condition/
   anthropometric metadata, selects the required columns, and writes one combined
   CSV. By default it reads from `Data - Second Skin/OpenSource_Dataset/Data/`
   and writes to `Data - Second Skin/Testing/` (both resolved relative to the
   repo, wherever it's cloned) — only edit the `data_dir` / `data_write_dir` /
   `data_req_dir` variables at the top of `main()` if your local layout differs.
   Then run:

   ```bash
   python data_extraction.py
   ```

4. **Point the scripts at your CSV.** `network_training.py`, `stats_cal.py`,
   `plot_error_plots.py`, and `joint_angle_plot.py` each build their `data_path`
   and checkpoint paths (e.g. `pae_model_file_path`) relative to the repo root
   (via a `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` at the top of
   each file) — edit those variables near the top of each file to point at your
   own CSV or checkpoint. In `network_training.py` and `stats_cal.py`, every such
   filename is marked with a `# TODO:` comment right above it — search for
   `TODO` in those files to find every spot that needs your own file name.

## Running the models

The main entry point is `network_training.py`. Open `main()` and set:

- `model_to_train` — one of `"PAE"`, `"MANN"`, `"MoETCNN"`, `"TCNN"`,
  `"FCNN_SW"`, or `"LSTM"`.
- `time_horizon_prediction` — prediction horizon in timesteps (1, 5, 20, 50, 80,
  or 100 in the paper's experiments).
- `data_path` — path to the combined CSV produced by `data_extraction.py`.
- For `"MANN"` / `"MoETCNN"`, `pae_model_file_path` — path to a PAE checkpoint
  trained beforehand (train `model_to_train = "PAE"` first, then point the MANN/
  MoE-TCN run at the saved `.pth` file).
- `log_wandB` — set to `False` if you don't want to log to Weights & Biases (or
  run `wandb login` / `wandb offline` first).

Then run:

```bash
python network_training.py
```

This builds the train/validation dataloaders, trains the selected model, saves
the weights and config to `Saved Models/`, and produces evaluation plots/
statistics for both the training and validation splits.

### Evaluation / figure-generation scripts

Once you have trained checkpoints saved under `Saved Models/`, the following
scripts (each with its own hardcoded checkpoint/data paths near the top of the
file — update them first) reproduce the paper's evaluation figures and tables:

- `stats_cal.py` — evaluates a checkpoint on held-out test-subject data at a
  chosen prediction horizon and prints per-joint MAE/STD/RMSE.
- `plot_error_plots.py` — computes per-joint MAE/STD/RMSE/R² at a fixed horizon
  across the PAE/TCNN/MANN/FCNN models.
- `joint_angle_plot.py` — plots predicted vs. ground-truth joint-angle
  trajectories over a fixed time window.
- `plot_results.py` — plots RMSE-vs-prediction-horizon curves from the result
  tables reported in the paper.

## Citations

If you use the Periodic Autoencoder implementation in `Models/PAE.py`, please
cite:

```bibtex
@article{starke2022deepphase,
  author    = {Starke, Sebastian and Mason, Ian and Komura, Taku},
  title     = {DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds},
  journal   = {ACM Transactions on Graphics},
  volume    = {41},
  number    = {4},
  articleno = {136},
  year      = {2022},
  doi       = {10.1145/3528223.3530178}
}
```

If you use the "Second Skin" dataset, please cite:

```bibtex
@article{casey2026secondskin,
  author  = {Casey, Ryan T. F. and Nuesslein, Christoph P. O. and Davenport, Felicia
             and Wheeler, Jason and Mazumdar, Anirban and Sawicki, Gregory and Young, Aaron J.},
  title   = {The Second Skin: A Wearable Sensor Suite That Enables Real-Time Human
             Biomechanics Tracking Through Deep Learning},
  journal = {IEEE Transactions on Biomedical Engineering},
  volume  = {73},
  number  = {2},
  pages   = {621--630},
  year    = {2026}
}
```

## Dependencies

Developed and tested with Python 3.10. Core dependencies:

- `torch` (2.6)
- `pandas` (2.2)
- `numpy` (1.26)
- `matplotlib` (3.10)
- `wandb` (0.19) — optional, only required if `log_wandB=True`

Install with:

```bash
pip install torch pandas numpy matplotlib wandb
```
