"""
PyTorch Dataset/Sampler used to feed the PAE, MANN, and TCN(N)-MoE models.

Builds fixed-length sliding windows over the combined dataset CSV, keeping every
window inside a single (subject, condition) trial, and splits each window into the
PAE (phase autoencoder) input, the motion-predictor input, and the future
motion-predictor target.

Author: Chinmay Shah
Institution: Institute for Human and Machine Cognition (IHMC) / University of West Florida (UWF)
"""

from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Iterable
from collections import defaultdict
import random


class GroupedSequenceDataset(Dataset):
    """
    Returns (PAE_in: [T,21], MANN_in: [T,43], MANN_out: [pred_len,20]).
    Windows never cross (subject, condition) boundaries.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        subject_col: str = "subject",
        condition_col: str = "condition",
        include_groups: Optional[Iterable[Tuple]] = None,
        stride: int = 1,
        input_start: int = 0,
        input_end: int = 46,
        output_start: int = 46,
        output_end: int = 66,
        dtype: torch.dtype = torch.float32,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = self.seq_len + self.pred_len
        self.indices = df.index.tolist()
        self.subject_col = subject_col
        self.condition_col = condition_col
        self.stride = stride
        self.input_start = input_start
        self.input_end = input_end
        self.output_start = output_start
        self.output_end = output_end
        self.dtype = dtype

        df = df.copy()

        # Optional group filtering (train/test split, etc.)
        if include_groups is not None:
            keys = set(include_groups)
            mask = [tuple(x) in keys for x in df[[subject_col, condition_col]].values.tolist()]
            df = df.loc[mask].copy()

        # Global normalization for prediction inputs
        inputs_all = df.iloc[:, self.input_start:self.input_end]
        self.input_mean = inputs_all.mean(axis=0)
        self.input_std = inputs_all.std(axis=0)
        self.input_std[self.input_std == 0] = 1
        
        # Global normalization for prediction inputs
        outputs_all = df.iloc[:, self.output_start:self.output_end]
        self.output_mean = outputs_all.mean(axis=0)
        self.output_std = outputs_all.std(axis=0)
        self.output_std[self.output_std == 0] = 1
        
        # Build groups: one DataFrame per (subject, condition)
        self.groups: Dict[Tuple, pd.DataFrame] = {}
        for key, g in df.groupby([subject_col, condition_col], sort=False):
            g = g.sort_index().reset_index(drop=True)
            
            PAE_input = g.iloc[:, 21:42].to_numpy(dtype=np.float64, copy=True)
            X = g.iloc[:, self.input_start:self.input_end].to_numpy(dtype=np.float64, copy=True)
            Y = g.iloc[:, self.output_start:self.output_end].to_numpy(dtype=np.float64, copy=True)

            # normalize once
            X = (X - self.input_mean.values) / self.input_std.values
            Y = (Y - self.output_mean.values) / self.output_std.values
            # PAE_input = X[:, 21:42] 
         
            self.groups[key] = {
                "Autoencoder_input": PAE_input.astype(np.float32, copy=False),  # [N,21]
                "Predictor_input": X.astype(np.float32, copy=False),          # [N,43]
                "Predictor_output": Y.astype(np.float32, copy=False),         # [N,20]
                "length": X.shape[0],
            }

        # Build window index: list of (group_key, start_idx)
        self.index: List[Tuple[Tuple, int]] = []
        for key, groups in self.groups.items():
            n = groups["length"]
            if n >= self.total_len:
                for s in range(0, n - self.total_len + 1, self.stride):
                    self.index.append((key, s))

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, i: int):
        key, s = self.index[i]
        group = self.groups[key]

        # Slice windows (no pandas, minimal allocation)
        # Shapes: [seq_len, F] / [pred_len, F]
        Autoencoder_input   = torch.tensor(group["Autoencoder_input"][s : s + self.seq_len].T)       
        Predictor_input  = torch.tensor(group["Predictor_input"][s : s + self.seq_len].T)      
        Predictor_output = torch.tensor(group["Predictor_output"][s + self.seq_len : s + self.total_len].T) 

        # Mean-center per feature for the autoencoder
        Autoencoder_input = Autoencoder_input - Autoencoder_input.mean(dim=1, keepdim=True)
        
        # Meta data to understand whats going on - not used currently
        meta = {
            "subject": key[0],
            "condition": key[1],
            "start_idx": s,
            "group_key": key,
        }
        # return Autoencoder_input, Predictor_input, Predictor_output, meta
        return Autoencoder_input, Predictor_input, Predictor_output


class GroupedBatchSampler(Sampler[List[int]]):
    """Keeps each batch within a single (subject, condition)."""

    def __init__(
        self,
        dataset: GroupedSequenceDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # bucket dataset indices by group
        buckets = defaultdict(list)
        for i, (key, _) in enumerate(dataset.index):
            buckets[key].append(i)
        self.buckets = dict(buckets)
        self.group_keys = list(self.buckets.keys())

    def __iter__(self):
        rng = random.Random()
        seed = 42
        rng.seed(seed)

        keys = self.group_keys[:]
        if self.shuffle:
            rng.shuffle(keys)

        for key in keys:
            idxs = self.buckets[key][:]
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        total = 0
        for _, idxs in self.buckets.items():
            n = len(idxs) // self.batch_size
            if not self.drop_last and (len(idxs) % self.batch_size):
                n += 1
            total += n
        return total
