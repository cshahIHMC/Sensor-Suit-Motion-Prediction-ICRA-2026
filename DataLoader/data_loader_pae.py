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
        time_col: str = "time",
        subject_col: str = "subject",
        condition_col: str = "condition",
        include_groups: Optional[Iterable[Tuple]] = None,
        stride: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = self.seq_len + self.pred_len
        self.time_col = time_col
        self.indices = df.index.tolist()
        self.subject_col = subject_col
        self.condition_col = condition_col
        self.stride = stride
        self.dtype = dtype

        df = df.copy()

        # Optional group filtering (train/test split, etc.)
        if include_groups is not None:
            keys = set(include_groups)
            mask = [tuple(x) in keys for x in df[[subject_col, condition_col]].values.tolist()]
            df = df.loc[mask].copy()

        # Global normalization for prediction inputs
        inputs_all = df.iloc[:, 0:43]
        self.input_mean = inputs_all.mean(axis=0)
        self.input_std = inputs_all.std(axis=0)
        self.input_std[self.input_std == 0] = 1
        
        # Global normalization for prediction inputs
        outputs_all = df.iloc[:, 43:63]
        self.output_mean = outputs_all.mean(axis=0)
        self.output_std = outputs_all.std(axis=0)
        self.output_std[self.output_std == 0] = 1
        
        # Build groups: one DataFrame per (subject, condition)
        self.groups: Dict[Tuple, pd.DataFrame] = {}
        for key, g in df.groupby([subject_col, condition_col], sort=False):
            # g = g.sort_values(self.time_col).reset_index(drop=True)
            g = g.sort_index().reset_index(drop=True)
            
            PAE_input = g.iloc[:, 0:21].to_numpy(dtype=np.float64, copy=True)
            X = g.iloc[:, 0:43].to_numpy(dtype=np.float64, copy=True)
            Y = g.iloc[:, 43:63].to_numpy(dtype=np.float64, copy=True)

            # normalize once
            X = (X - self.input_mean.values) / self.input_std.values
            Y = (Y - self.output_mean.values) / self.output_std.values
         
            self.groups[key] = {
                "pae_in": PAE_input.astype(np.float32, copy=False),  # [N,21]
                "mann_in": X.astype(np.float32, copy=False),          # [N,43]
                "mann_out": Y.astype(np.float32, copy=False),         # [N,20]
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
        PAE_input   = torch.tensor(group["pae_in"][s : s + self.seq_len].T)         # [21,Seq_length]
        MANN_input  = torch.tensor(group["mann_in"][s : s + self.seq_len].T)            # [43,Seq_length]
        MANN_output = torch.tensor(group["mann_out"][s + self.seq_len : s + self.total_len].T)    # [20,Pred_length]

        # Mean-center per feature
        PAE_input = PAE_input - PAE_input.mean(dim=1, keepdim=True)

        meta = {
            "subject": key[0],
            "condition": key[1],
            "start_idx": s,
            "group_key": key,
        }
        # return PAE_input, MANN_input, MANN_output, meta
        return PAE_input, MANN_input, MANN_output


class GroupedBatchSampler(Sampler[List[int]]):
    """Keeps each batch within a single (subject, condition)."""

    def __init__(
        self,
        dataset: GroupedSequenceDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        # bucket dataset indices by group
        buckets = defaultdict(list)
        for i, (key, _) in enumerate(dataset.index):
            buckets[key].append(i)
        self.buckets = dict(buckets)
        self.group_keys = list(self.buckets.keys())

    def __iter__(self):
        rng = random.Random()
        if self.generator is not None:
            # seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.generator).item())
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
