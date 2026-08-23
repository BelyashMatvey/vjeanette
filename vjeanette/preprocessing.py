#!/usr/bin/env python
# coding: utf-8

"""
Preprocessing pipeline for VDJ sequence data.

This script processes IgBLAST output files, aggregates Ig/TCR data,
builds a dataset with positive/negative samples, and serializes it
for model training.
"""

import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

# Constants
MAX_LEN = 512

CHAR_TO_IDX = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'N': 4,
    'P': 5
}

USECOLS = [
    'sequence',
    'cdr3_start', 'cdr3_end',
    'v_sequence_start', 'v_sequence_end',
    'j_sequence_start', 'j_sequence_end',
    'v_score', 'v_identity',
    'j_score', 'j_identity',
    'locus', 'sequence_id', 'rev_comp'
]


def load_and_aggregate(ig_path, tcr_path, chunk_size=1_000_000, verbose=True):
    """
    Load and aggregate Ig and TCR data from IgBLAST output files.
    
    Args:
        ig_path: Path to IgBLAST Ig output file
        tcr_path: Path to IgBLAST TCR output file
        chunk_size: Number of rows to process at once
        verbose: Print progress information
    
    Returns:
        Aggregated DataFrame with combined Ig/TCR annotations
    """
    if verbose:
        print("=" * 60)
        print("STARTING IG/TCR AGGREGATION")
        print("=" * 60)

    def process_file(path):
        """Process a single IgBLAST output file."""
        if verbose:
            print(f"Processing file: {path}")

        dfs = []
        total_chunks = 0
        total_rows = 0

        # Create iterator with progress bar if verbose
        chunks = pd.read_csv(
            path,
            usecols=USECOLS,
            delimiter="\t",
            chunksize=chunk_size
        )
        
        if verbose:
            chunks = tqdm(chunks, desc=f"Processing {os.path.basename(path)}")

        for chunk in chunks:
            total_chunks += 1
            total_rows += len(chunk)

            if verbose and total_chunks % 10 == 0:
                print(f"[{os.path.basename(path)}] "
                      f"Chunk {total_chunks} | rows={len(chunk):,}")

            # Get best V annotation per sequence
            v = (
                chunk.sort_values("v_identity", ascending=False)
                .drop_duplicates("sequence_id")
                [["sequence_id", "v_sequence_start", "v_sequence_end",
                  "v_score", "v_identity"]]
            )

            # Get best J annotation per sequence
            j = (
                chunk.sort_values("j_identity", ascending=False)
                .drop_duplicates("sequence_id")
                [["sequence_id", "j_sequence_start", "j_sequence_end",
                  "j_score", "j_identity"]]
            )

            # Get CDR3 annotation per sequence (only where present)
            cdr3 = (
                chunk.dropna(subset=["cdr3_start"])
                .drop_duplicates("sequence_id")
                [["sequence_id", "cdr3_start", "cdr3_end"]]
            )

            # Get sequence data
            seq = (
                chunk[["sequence_id", "sequence"]]
                .drop_duplicates("sequence_id")
            )

            # Merge all annotations
            df = seq.merge(v, how="left", on="sequence_id")
            df = df.merge(j, how="left", on="sequence_id")
            df = df.merge(cdr3, how="left", on="sequence_id")

            dfs.append(df)

        if verbose:
            print(f"Finished {path} | chunks={total_chunks} | "
                  f"rows={total_rows:,}")

        result = pd.concat(dfs)
        
        if verbose:
            print(f"Aggregated dataframe shape: {result.shape}")

        return result

    # Process both files
    ig = process_file(ig_path)
    tcr = process_file(tcr_path)

    if verbose:
        print("Computing quality scores...")

    def score(df):
        """Compute quality score for ranking annotations."""
        return (
            df.v_score.fillna(0) +
            df.j_score.fillna(0) +
            2 * (df.v_identity.fillna(0) + df.j_identity.fillna(0))
        )

    ig["score"] = score(ig)
    tcr["score"] = score(tcr)

    if verbose:
        print("Merging IG/TCR datasets...")

    # Merge Ig and TCR data
    merged = pd.merge(
        ig,
        tcr,
        on="sequence_id",
        how="outer",
        suffixes=("_ig", "_tcr")
    )

    if verbose:
        print(f"Merged shape: {merged.shape}")

    # Choose better annotation between Ig and TCR
    choose_ig = (
        merged.score_ig.fillna(-1) >=
        merged.score_tcr.fillna(-1)
    )

    # Build final dataset with best annotations
    final = pd.DataFrame({
        "sequence": np.where(
            choose_ig,
            merged.sequence_ig,
            merged.sequence_tcr
        ),
        "sequence_id": merged.sequence_id,
        "v_sequence_start": np.where(
            choose_ig,
            merged.v_sequence_start_ig,
            merged.v_sequence_start_tcr
        ),
        "v_sequence_end": np.where(
            choose_ig,
            merged.v_sequence_end_ig,
            merged.v_sequence_end_tcr
        ),
        "j_sequence_start": np.where(
            choose_ig,
            merged.j_sequence_start_ig,
            merged.j_sequence_start_tcr
        ),
        "j_sequence_end": np.where(
            choose_ig,
            merged.j_sequence_end_ig,
            merged.j_sequence_end_tcr
        ),
        "cdr3_start": np.where(
            choose_ig,
            merged.cdr3_start_ig,
            merged.cdr3_start_tcr
        ),
        "cdr3_end": np.where(
            choose_ig,
            merged.cdr3_end_ig,
            merged.cdr3_end_tcr
        ),
        "best_v_identity": np.where(
            choose_ig,
            merged.v_identity_ig,
            merged.v_identity_tcr
        ),
        "best_j_identity": np.where(
            choose_ig,
            merged.j_identity_ig,
            merged.j_identity_tcr
        ),
        "best_v_score": np.where(
            choose_ig,
            merged.v_score_ig,
            merged.v_score_tcr
        ),
        "best_j_score": np.where(
            choose_ig,
            merged.j_score_ig,
            merged.j_score_tcr
        ),
    })

    final["has_v"] = final.v_sequence_start.notna()
    final["has_j"] = final.j_sequence_start.notna()

    if verbose:
        print("=" * 60)
        print("FINAL DATASET SUMMARY")
        print("=" * 60)
        print(f"Final shape: {final.shape}")
        print(f"Has V: {final.has_v.sum():,}")
        print(f"Has J: {final.has_j.sum():,}")

    return final


def make_vector(start, end, max_len=MAX_LEN):
    """
    Create a binary vector with ones between start and end positions.
    
    Args:
        start: Starting position (1-indexed)
        end: Ending position (exclusive)
        max_len: Maximum sequence length
    
    Returns:
        Binary tensor of shape (max_len,)
    """
    v = torch.zeros(max_len, dtype=torch.float32)

    if pd.isna(start) or pd.isna(end):
        return v

    start = int(start) - 1
    end = int(end)

    start = max(0, start)
    end = min(max_len, end)

    if start < end:
        v[start:end] = 1

    return v


class VDJDataset(Dataset):
    """PyTorch Dataset for VDJ sequence data."""

    def __init__(
        self,
        ig_path,
        tcr_path,
        neg_ratio=1,
        v_identity=80,
        v_score=50,
        j_identity=90,
        j_score=40,
        max_v_pos=50000,
        max_j_pos=50000,
        verbose=True,
        random_state=42
    ):
        """
        Initialize VDJ dataset.
        
        Args:
            ig_path: Path to IgBLAST Ig output
            tcr_path: Path to IgBLAST TCR output
            neg_ratio: Ratio of negative to positive samples
            v_identity: V gene identity threshold for positive
            v_score: V gene score threshold for positive
            j_identity: J gene identity threshold for positive
            j_score: J gene score threshold for positive
            max_v_pos: Maximum number of V positive samples
            max_j_pos: Maximum number of J positive samples
            verbose: Print progress information
            random_state: Random seed for reproducibility
        """
        if verbose:
            print("=" * 60)
            print("BUILDING DATASET")
            print("=" * 60)

        # Load and aggregate data
        df = load_and_aggregate(ig_path, tcr_path, verbose=verbose)

        if verbose:
            print("Applying V/J positive filters...")

        # Define positive masks
        v_pos_mask = (
            df.has_v &
            (df.best_v_identity > v_identity) &
            (df.best_v_score > v_score)
        )

        j_pos_mask = (
            df.has_j &
            (df.best_j_identity > j_identity) &
            (df.best_j_score > j_score)
        )

        cdr3_pos_mask = v_pos_mask & j_pos_mask

        if verbose:
            print("\nLABEL DISTRIBUTION")
            print("-" * 60)
            print(f"V positive:    {v_pos_mask.sum():,}")
            print(f"J positive:    {j_pos_mask.sum():,}")
            print(f"CDR3 positive: {cdr3_pos_mask.sum():,}")

            print("\n2x2 V/J distribution")
            print("-" * 60)
            
            vj_pp = (v_pos_mask & j_pos_mask).sum()
            vj_pn = (v_pos_mask & ~j_pos_mask).sum()
            vj_np = (~v_pos_mask & j_pos_mask).sum()
            vj_nn = (~v_pos_mask & ~j_pos_mask).sum()

            print(f"V+ J+ : {vj_pp:,}")
            print(f"V+ J- : {vj_pn:,}")
            print(f"V- J+ : {vj_np:,}")
            print(f"V- J- : {vj_nn:,}")

        # Sample positive samples
        v_pos = df[v_pos_mask].copy()
        if max_v_pos is not None and len(v_pos) > max_v_pos:
            v_pos = v_pos.sample(n=max_v_pos, random_state=random_state)

        j_pos = df[j_pos_mask].copy()
        if max_j_pos is not None and len(j_pos) > max_j_pos:
            j_pos = j_pos.sample(n=max_j_pos, random_state=random_state)

        # Sample negative samples
        v_neg_candidates = df[~v_pos_mask]
        j_neg_candidates = df[~j_pos_mask]

        n_v_neg = min(len(v_pos) * neg_ratio, len(v_neg_candidates))
        v_neg = v_neg_candidates.sample(n=n_v_neg, random_state=random_state)

        n_j_neg = min(len(j_pos) * neg_ratio, len(j_neg_candidates))
        j_neg = j_neg_candidates.sample(n=n_j_neg, random_state=random_state + 1)

        # Combine all samples
        data = pd.concat([v_pos, j_pos, v_neg, j_neg], ignore_index=True)
        data = data.drop_duplicates(subset=["sequence"]).reset_index(drop=True)

        self.data = data

        # Store masks for the final dataset
        self.v_pos_mask = (
            self.data.has_v &
            (self.data.best_v_identity > v_identity) &
            (self.data.best_v_score > v_score)
        )

        self.j_pos_mask = (
            self.data.has_j &
            (self.data.best_j_identity > j_identity) &
            (self.data.best_j_score > j_score)
        )

        self.cdr3_pos_mask = self.v_pos_mask & self.j_pos_mask

        if verbose:
            print("\n" + "=" * 60)
            print("DATASET READY")
            print("=" * 60)
            print(f"Total samples: {len(self.data):,}")
            print(f"V positives: {self.v_pos_mask.sum():,}")
            print(f"J positives: {self.j_pos_mask.sum():,}")
            print(f"CDR3 positives: {self.cdr3_pos_mask.sum():,}")
            print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        if idx % 100000 == 0 and idx > 0:
            print(f"Processed {idx:,} samples")

        row = self.data.iloc[idx]

        # Encode sequence
        seq = row.sequence[:MAX_LEN]
        seq = seq + "P" * (MAX_LEN - len(seq))
        X = torch.tensor([CHAR_TO_IDX[c] for c in seq], dtype=torch.long)

        # Create labels: [CDR3, V, J]
        y = torch.zeros((3, MAX_LEN), dtype=torch.float32)

        # CDR3 label
        if self.cdr3_pos_mask.iloc[idx]:
            y[0] = make_vector(row.cdr3_start, row.cdr3_end)

        # V label
        if self.v_pos_mask.iloc[idx]:
            y[1] = make_vector(row.v_sequence_start, row.v_sequence_end)

        # J label
        if self.j_pos_mask.iloc[idx]:
            y[2] = make_vector(row.j_sequence_start, row.j_sequence_end)

        return X, y


def dataset_to_pt(dataset, out_path, verbose=True):
    """
    Convert dataset to PyTorch tensors and save to file.
    
    Args:
        dataset: VDJDataset instance
        out_path: Output file path
        verbose: Print progress information
    """
    if verbose:
        print("=" * 60)
        print("SERIALIZING DATASET")
        print("=" * 60)

    X_list = []
    y_list = []
    total = len(dataset)

    for i in tqdm(range(total), desc="Tensorizing", disable=not verbose):
        if verbose and i % 50000 == 0:
            print(f"Tensorizing {i:,}/{total:,}")

        x, y = dataset[i]
        X_list.append(x)
        y_list.append(y)

    if verbose:
        print("Stacking tensors...")

    X = torch.stack(X_list)
    y = torch.stack(y_list)

    if verbose:
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Saving dataset -> {out_path}")

    torch.save({"X": X, "y": y}, out_path)

    if verbose:
        print("=" * 60)
        print("DATASET SAVED SUCCESSFULLY")
        print("=" * 60)


def combine_datasets(input_paths, output_path, verbose=True):
    """
    Combine multiple saved dataset files into one.
    
    Args:
        input_paths: List of paths to .pt files
        output_path: Output path for combined dataset
        verbose: Print progress information
    """
    if verbose:
        print("=" * 60)
        print("COMBINING DATASETS")
        print("=" * 60)

    X_list = []
    y_list = []

    for path in input_paths:
        if verbose:
            print(f"Loading: {path}")

        data = torch.load(path)
        X_list.append(data['X'])
        y_list.append(data['y'])

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    if verbose:
        print(f"Combined X shape: {X.shape}")
        print(f"Combined y shape: {y.shape}")
        print(f"Saving to: {output_path}")

    torch.save({'X': X, 'y': y}, output_path)

    if verbose:
        print("=" * 60)
        print("DATASETS COMBINED SUCCESSFULLY")
        print("=" * 60)