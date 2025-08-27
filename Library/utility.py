import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import numpy as np

### Contains all helper functions

# Check if the GPU is available and put the object on the GPU
def ToDevice(x):
    return x.cuda() if torch.cuda.is_available() else x

# To detach from the GPU
def Item(value):
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
                    condition_col: str = "condition",
                    time_col: str = "time") -> Dict[Tuple[str,str], pd.DataFrame]:
    groups: Dict[Tuple[str, str], pd.DataFrame] = {}
    use_time = time_col in df.columns
    for key, g in df.groupby([subject_col, condition_col], sort=False):
        g = g.sort_values(time_col) if use_time else g.sort_index()
        groups[key] = g.reset_index(drop=True)
    return groups

# Splitting According to conditions or subject
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
