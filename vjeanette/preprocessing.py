# preprocessing.py

import os
import torch
import pandas as pd
import numpy as np
import logging
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LEN = 512

USECOLS = [
    'sequence',
    'cdr3_start','cdr3_end',
    'v_sequence_start','v_sequence_end',
    'j_sequence_start','j_sequence_end',
    'v_score','v_identity',
    'j_score','j_identity',
    'locus','sequence_id', 'rev_comp'
]

CHAR_TO_IDX = {'A':0,'C':1,'G':2,'T':3,'N':4,'P':5}
IDX_TO_CHAR = ['A','C','G','T','N','P']

def load_and_aggregate(ig_path, tcr_path):

    logger.info("=" * 60)
    logger.info("STARTING IG/TCR AGGREGATION")
    logger.info("=" * 60)

    CHUNK = 1_000_000

    def process_file(path):

        logger.info(f"Processing file: {path}")

        dfs = []
        total_chunks = 0
        total_rows = 0

        for chunk_id, chunk in enumerate(
            pd.read_csv(
                path,
                usecols=USECOLS,
                delimiter="\t",
                chunksize=CHUNK
            )
        ):

            total_chunks += 1
            total_rows += len(chunk)

            logger.info(
                f"[{os.path.basename(path)}] "
                f"Chunk {chunk_id+1} | rows={len(chunk):,}"
            )

            v = (
                chunk.sort_values("v_identity", ascending=False)
                .drop_duplicates("sequence_id")
                [[
                    "sequence_id",
                    "v_sequence_start",
                    "v_sequence_end",
                    "v_score",
                    "v_identity"
                ]]
            )

            j = (
                chunk.sort_values("j_identity", ascending=False)
                .drop_duplicates("sequence_id")
                [[
                    "sequence_id",
                    "j_sequence_start",
                    "j_sequence_end",
                    "j_score",
                    "j_identity"
                ]]
            )

            cdr3 = (
                chunk.dropna(subset=["cdr3_start"])
                .drop_duplicates("sequence_id")
                [[
                    "sequence_id",
                    "cdr3_start",
                    "cdr3_end"
                ]]
            )

            seq = (
                chunk[["sequence_id","sequence"]]
                .drop_duplicates("sequence_id")
            )

            df = seq.merge(v, how="left", on="sequence_id")
            df = df.merge(j, how="left", on="sequence_id")
            df = df.merge(cdr3, how="left", on="sequence_id")

            dfs.append(df)

        logger.info(
            f"Finished {path} | "
            f"chunks={total_chunks} | "
            f"rows={total_rows:,}"
        )

        result = pd.concat(dfs)

        logger.info(
            f"Aggregated dataframe shape: {result.shape}"
        )

        return result

    ig = process_file(ig_path)
    tcr = process_file(tcr_path)

    logger.info("Computing quality scores...")

    def score(df):
        return (
            df.v_score.fillna(0) +
            df.j_score.fillna(0) +
            2*(df.v_identity.fillna(0) + df.j_identity.fillna(0))
        )

    ig["score"] = score(ig)
    tcr["score"] = score(tcr)

    logger.info("Merging IG/TCR datasets...")

    merged = pd.merge(
        ig,
        tcr,
        on="sequence_id",
        how="outer",
        suffixes=("_ig","_tcr")
    )

    logger.info(f"Merged shape: {merged.shape}")

    choose_ig = (
        merged.score_ig.fillna(-1)
        >=
        merged.score_tcr.fillna(-1)
    )

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

    logger.info("=" * 60)
    logger.info("FINAL DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Final shape: {final.shape}")
    logger.info(f"Has V: {final.has_v.sum():,}")
    logger.info(f"Has J: {final.has_j.sum():,}")

    return final


class VDJDataset(Dataset):

    def __init__(self, ig_path, tcr_path, neg_ratio=1):

        logger.info("=" * 60)
        logger.info("BUILDING DATASET")
        logger.info("=" * 60)

        df = load_and_aggregate(ig_path, tcr_path)

        logger.info("Applying positive filters...")

        pos_mask = (
            df.has_v &
            df.has_j &
            (df.best_v_identity > 80) &
            (df.best_j_identity > 88) &
            (df.best_v_score > 50) &
            (df.best_j_score > 40)
        )

        pos = df[pos_mask].copy()

        logger.info(f"Positive samples: {len(pos):,}")

        logger.info("Sampling negatives...")

        neg = df[
            (~df.has_v) &
            (~df.has_j)
        ].sample(
            n=min(len(pos)*neg_ratio, len(df)),
            random_state=42
        )

        logger.info(f"Negative samples: {len(neg):,}")

        self.data = pd.concat([pos, neg]).reset_index(drop=True)
        self.pos_len = len(pos)

        logger.info("=" * 60)
        logger.info("DATASET READY")
        logger.info("=" * 60)
        logger.info(f"Total samples: {len(self.data):,}")
        logger.info(f"Positive ratio: {self.pos_len / len(self.data):.3f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if idx % 100000 == 0 and idx > 0:
            logger.info(f"Processed {idx:,} samples")

        row = self.data.iloc[idx]

        seq = row.sequence[:MAX_LEN]
        seq = seq + "P"*(MAX_LEN-len(seq))

        X = torch.tensor(
            [CHAR_TO_IDX[c] for c in seq]
        )

        y = torch.zeros((3,MAX_LEN))

        if idx < self.pos_len:
            y[0] = make_vector(
                row.cdr3_start,
                row.cdr3_end
            )

            y[1] = make_vector(
                row.v_sequence_start,
                row.v_sequence_end
            )

            y[2] = make_vector(
                row.j_sequence_start,
                row.j_sequence_end
            )

        return X, y


def make_vector(start, end):

    v = torch.zeros(MAX_LEN)

    if pd.isna(start) or pd.isna(end):
        return v

    v[int(start)-1:int(end)] = 1

    return v


def dataset_to_pt(dataset, out_path):

    logger.info("=" * 60)
    logger.info("SERIALIZING DATASET")
    logger.info("=" * 60)

    X_list = []
    y_list = []

    total = len(dataset)

    for i in range(total):

        if i % 50000 == 0:
            logger.info(
                f"Tensorizing {i:,}/{total:,}"
            )

        x, y = dataset[i]

        X_list.append(x)
        y_list.append(y)

    logger.info("Stacking tensors...")

    X = torch.stack(X_list)
    y = torch.stack(y_list)

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")

    logger.info(f"Saving dataset -> {out_path}")

    torch.save(
        {'X': X, 'y': y},
        out_path
    )

    logger.info("=" * 60)
    logger.info("DATASET SAVED SUCCESSFULLY")
    logger.info("=" * 60)