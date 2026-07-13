"""
Scratch/testing utilities: standalone copies of the subject/condition grouping
helpers and an alternate `GroupedSequenceDataset` (single-tensor window output,
optional insole-channel normalization) used while iterating on the DataLoader
design in DataLoader/data_loader_pae.py. Not imported by the main training
pipeline.

Author: Chinmay Shah
Institution: Institute for Human and Machine Cognition (IHMC) / University of West Florida (UWF)
"""

import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Optional, Tuple, Dict, Iterable
import torch
from collections import defaultdict
import random
import os

# Repo root (folder this file lives in) - used so the default paths below work
# regardless of where the repository is cloned.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_pairs(df: pd.DataFrame,
                  subject_col: str = "subject",
                  condition_col: str = "condition") -> Tuple[List[str], List[str], List[Tuple[str,str]]]:
    """Return the unique subjects, unique conditions, and unique (subject, condition) pairs in `df`."""
    subjects   = df[subject_col].dropna().unique().tolist()
    conditions = df[condition_col].dropna().unique().tolist()
    pairs = list(
        df[[subject_col, condition_col]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    return subjects, conditions, pairs

def make_group_dict(df: pd.DataFrame,
                    subject_col: str = "subject",
                    condition_col: str = "condition",
                    time_col: str = "time") -> Dict[Tuple[str,str], pd.DataFrame]:
    groups: Dict[Tuple[str, str], pd.DataFrame] = {}
    use_time = time_col in df.columns
    for key, g in df.groupby([subject_col, condition_col], sort=False):
        g = g.sort_values(time_col) if use_time else g.sort_index()
        groups[key] = g.reset_index(drop=True)
    return groups


def summarize_groups(groups: Dict[Tuple[str,str], pd.DataFrame]) -> pd.DataFrame:
    rows = [{"subject": s, "condition": c, "rows": len(g)} for (s, c), g in groups.items()]
    return pd.DataFrame(rows).sort_values(["subject", "condition"]).reset_index(drop=True)


def windows_per_group(groups: Dict[Tuple[str,str], pd.DataFrame],
                      seq_len: int,
                      stride: int) -> pd.DataFrame:
    recs = []
    for (s, c), g in groups.items():
        n = len(g)
        w = 0
        if n >= seq_len:
            w = (n - seq_len) // stride + 1
        recs.append({"subject": s, "condition": c, "rows": n, "windows": w})
    return pd.DataFrame(recs).sort_values(["subject","condition"]).reset_index(drop=True)

# def split_by_subject(df: pd.DataFrame,
#                      subject_col: str = "subject",
#                      train_frac=0.7, val_frac=0.15, seed=42) -> Tuple[set, set, set]:
#     rng = np.random.default_rng(seed)
#     subjects = sorted(df[subject_col].unique().tolist())
#     rng.shuffle(subjects)
#     n = len(subjects)
#     n_train = int(train_frac * n)
#     n_val   = int(val_frac * n)
#     train = set(subjects[:n_train])
#     val   = set(subjects[n_train:n_train+n_val])
#     test  = set(subjects[n_train+n_val:])
#     return train, val, test

# def pairs_for_subjects(pairs: List[Tuple[str,str]], subjects: set) -> List[Tuple[str,str]]:
#     return [(s, c) for (s, c) in pairs if s in subjects]

# def split_by_pairs(pairs: List[Tuple[str,str]], seed=42, val_frac=0.15, test_frac=0.15):
#     """Optional alternative: split directly by (subject, condition) pairs."""
#     rng = np.random.default_rng(seed)
#     pairs = pairs.copy()
#     rng.shuffle(pairs)
#     n = len(pairs)
#     n_val  = int(val_frac * n)
#     n_test = int(test_frac * n)
#     val   = set(pairs[:n_val])
#     test  = set(pairs[n_val:n_val+n_test])
#     train = set(pairs[n_val+n_test:])
#     return train, val, test

class GroupedSequenceDataset(Dataset):
    """
    Returns (X_centered: [F, T], meta) windows where groups = (subject, condition).
    - gyro columns: df.iloc[:, 0:21]
    - insole columns: df.iloc[:, 42:48]  (column-wise normalized)
    - windows never cross (subject, condition) boundaries

    Normalization:
      - By default, insole mean/std are computed over the (optionally filtered) df you pass.
      - You can freeze stats by passing frozen_ins_mean/frozen_ins_std from your train dataset.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        time_col: str = "time",
        subject_col: str = "subject",
        condition_col: str = "condition",
        include_groups: Optional[Iterable[Tuple]] = None,
        stride: int = 1,
        dtype: torch.dtype = torch.float32,
        frozen_ins_mean: Optional[pd.Series] = None,
        frozen_ins_std: Optional[pd.Series] = None,
    ):
        self.seq_len = seq_len
        self.time_col = time_col
        self.subject_col = subject_col
        self.condition_col = condition_col
        self.stride = stride
        self.dtype = dtype

        df = df.copy()

        # Optional group filtering (e.g., just train pairs)
        if include_groups is not None:
            keys = set(include_groups)
            mask = [tuple(x) in keys for x in df[[subject_col, condition_col]].values.tolist()]
            df = df.loc[mask].copy()

        # Build groups per (subject, condition)
        self.groups: Dict[Tuple[Tuple[str,str], pd.DataFrame]] = {}
        use_time = time_col in df.columns
        for key, g in df.groupby([subject_col, condition_col], sort=False):
            g = g.sort_values(time_col) if use_time else g.sort_index()
            self.groups[key] = g.reset_index(drop=True)

        # Insole normalization stats
        # if frozen_ins_mean is not None and frozen_ins_std is not None:
        #     self.ins_mean = frozen_ins_mean
        #     self.ins_std  = frozen_ins_std
        # else:
        #     insole_all = df.iloc[:, self.insole_slice]
        #     self.ins_mean = insole_all.mean()
        #     self.ins_std  = insole_all.std()
        #     self.ins_std[self.ins_std == 0] = 1

        # Build index of windows: list of (group_key, start_idx)
        self.index: List[Tuple[Tuple[str,str], int]] = []
        for key, g in self.groups.items():
            n = len(g)
            if n >= self.seq_len:
                for s in range(0, n - self.seq_len + 1, self.stride):
                    self.index.append((key, s))

    def __len__(self):
        return len(self.index)

    def _extract_block(self, g: pd.DataFrame, start: int, end: int) -> np.ndarray:
        gyro = g.iloc[start:end, self.gyro_slice].to_numpy(dtype=np.float64, copy=True)
        ins  = g.iloc[start:end, self.insole_slice].to_numpy(dtype=np.float64, copy=True)
        ins  = (ins - self.ins_mean.values) / self.ins_std.values
        X    = np.hstack([gyro, ins])  # [T, F]
        return X

    def __getitem__(self, i: int):
        key, s = self.index[i]
        g = self.groups[key]

        X = self._extract_block(g, s, s + self.seq_len)          # [T, F]
        X = torch.tensor(X, dtype=self.dtype).transpose(0, 1)     # [F, T]

        # mean-center per feature across time
        X = X - X.mean(dim=1, keepdim=True)

        meta = {
            "subject": key[0],
            "condition": key[1],
            "start_idx": s,
            "group_key": key,
        }
        return X, meta


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

        buckets = defaultdict(list)
        for i, (key, _) in enumerate(dataset.index):
            buckets[key].append(i)
        self.buckets = dict(buckets)
        self.group_keys = list(self.buckets.keys())

    def __iter__(self):
        rng = random.Random()
        if self.generator is not None:
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.generator).item())
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
    

def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "Data", "all_subjects_req_sim_data.csv"))
    print(df.shape)
    
    subjects, conditions, pairs = extract_pairs(df)
    print(len(subjects))
    print(len(conditions))
    print(len(pairs))
    groups = make_group_dict(df)
    print(groups)
    row_list = summarize_groups(groups)
    print(row_list)
    # windows = windows_per_group(groups, 200, 1)
    
    



if __name__ == "__main__":
    raise SystemExit(main())
    