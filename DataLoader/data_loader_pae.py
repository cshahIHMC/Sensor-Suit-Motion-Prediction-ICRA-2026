from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Iterable
from collections import defaultdict
import random


class GroupedSequenceDataset(Dataset):
    """
    Returns (X_centered: [F, T], meta) windows where groups = (subject, condition).
    - gyro columns: df.iloc[:, 0:21]
    - insole columns: df.iloc[:, 42:48]  (column-wise normalized globally)
    - windows never cross (subject, condition) boundaries
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
    ):
        self.seq_len = seq_len
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

        # Build groups: one DataFrame per (subject, condition)
        self.groups: Dict[Tuple, pd.DataFrame] = {}
        for key, g in df.groupby([subject_col, condition_col], sort=False):
            # g = g.sort_values(self.time_col).reset_index(drop=True)
            g = g.sort_index().reset_index(drop=True)
            self.groups[key] = g

        # # Global insole normalization stats (over this df)
        # insole_all = df.iloc[:, self.insole_slice]
        # self.ins_mean = insole_all.mean()
        # self.ins_std = insole_all.std()
        # self.ins_std[self.ins_std == 0] = 1

        # Build window index: list of (group_key, start_idx)
        self.index: List[Tuple[Tuple, int]] = []
        for key, g in self.groups.items():
            n = len(g)
            if n >= self.seq_len:
                for s in range(0, n - self.seq_len + 1, self.stride):
                    self.index.append((key, s))

    def __len__(self):
        return len(self.index)

    def _extract_block(self, g: pd.DataFrame, start: int, end: int) -> np.ndarray:
        gyro = g.iloc[start:end, 0:21].to_numpy(dtype=np.float64, copy=True)
        # ins = g.iloc[start:end, self.insole_slice].to_numpy(dtype=np.float64, copy=True)
        # ins = (ins - self.ins_mean.values) / self.ins_std.values
        # X = np.hstack([gyro, ins])  # shape: [T, F]
        return gyro

    def __getitem__(self, i: int):
        key, s = self.index[i]
        g = self.groups[key]

        X = self._extract_block(g, s, s + self.seq_len)           # [T, F]
        X = torch.tensor(X, dtype=self.dtype).transpose(0, 1)     # [F, T]

        # Mean-center per feature
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

        # bucket dataset indices by group
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
