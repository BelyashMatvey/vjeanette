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


def load_and_aggregate(ig_path, tcr_path):
    logger.info("FAST IG/TCR aggregation")
    CHUNK = 1_000_000

    def process_file(path):
        dfs = []
        for chunk in pd.read_csv(
            path,
            usecols=USECOLS,
            delimiter="\t",
            chunksize=CHUNK
        ):
            v = (
                chunk.sort_values("v_identity", ascending=False)
                .drop_duplicates("sequence_id")
                [["sequence_id","v_sequence_start","v_sequence_end",
                  "v_score","v_identity"]]
            )

            j = (
                chunk.sort_values("j_identity", ascending=False)
                .drop_duplicates("sequence_id")
                [["sequence_id","j_sequence_start","j_sequence_end",
                  "j_score","j_identity"]]
            )

            cdr3 = (
                chunk.dropna(subset=["cdr3_start"])
                .drop_duplicates("sequence_id")
                [["sequence_id","cdr3_start","cdr3_end"]]
            )

            seq = chunk[["sequence_id","sequence"]].drop_duplicates("sequence_id")

            df = seq.merge(v, how="left", on="sequence_id")
            df = df.merge(j, how="left", on="sequence_id")
            df = df.merge(cdr3, how="left", on="sequence_id")

            dfs.append(df)

        return pd.concat(dfs)

    ig = process_file(ig_path)
    tcr = process_file(tcr_path)

    def score(df):
        return (
            df.v_score.fillna(0) +
            df.j_score.fillna(0) +
            2*(df.v_identity.fillna(0) + df.j_identity.fillna(0))
        )

    ig["score"] = score(ig)
    tcr["score"] = score(tcr)

    merged = pd.merge(
        ig, tcr,
        on="sequence_id",
        how="outer",
        suffixes=("_ig","_tcr")
    )

    choose_ig = merged.score_ig.fillna(-1) >= merged.score_tcr.fillna(-1)

    final = pd.DataFrame({
        "sequence": np.where(choose_ig, merged.sequence_ig, merged.sequence_tcr),
        "sequence_id": merged.sequence_id,
        "v_sequence_start": np.where(choose_ig, merged.v_sequence_start_ig, merged.v_sequence_start_tcr),
        "v_sequence_end": np.where(choose_ig, merged.v_sequence_end_ig, merged.v_sequence_end_tcr),
        "j_sequence_start": np.where(choose_ig, merged.j_sequence_start_ig, merged.j_sequence_start_tcr),
        "j_sequence_end": np.where(choose_ig, merged.j_sequence_end_ig, merged.j_sequence_end_tcr),
        "cdr3_start": np.where(choose_ig, merged.cdr3_start_ig, merged.cdr3_start_tcr),
        "cdr3_end": np.where(choose_ig, merged.cdr3_end_ig, merged.cdr3_end_tcr),
        "best_v_identity": np.where(choose_ig, merged.v_identity_ig, merged.v_identity_tcr),
        "best_j_identity": np.where(choose_ig, merged.j_identity_ig, merged.j_identity_tcr),
        "best_v_score": np.where(choose_ig, merged.v_score_ig, merged.v_score_tcr),
        "best_j_score": np.where(choose_ig, merged.j_score_ig, merged.j_score_tcr),
    })

    final["has_v"] = final.v_sequence_start.notna()
    final["has_j"] = final.j_sequence_start.notna()

    return final


class VDJDataset(Dataset):
    def __init__(self, ig_path, tcr_path, neg_ratio=1):
        df = load_and_aggregate(ig_path, tcr_path)

        pos_mask = (
            df.has_v &
            df.has_j &
            (df.best_v_identity > 80) &
            (df.best_j_identity > 88) &
            (df.best_v_score > 50) &
            (df.best_j_score > 40)
        )

        pos = df[pos_mask].copy()
        neg = df[(~df.has_v) & (~df.has_j)].sample(
            n=min(len(pos)*neg_ratio, len(df)),
            random_state=42
        )

        self.data = pd.concat([pos, neg]).reset_index(drop=True)
        self.pos_len = len(pos)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        seq = row.sequence[:MAX_LEN]
        seq = seq + "P"*(MAX_LEN-len(seq))

        X = torch.tensor([CHAR_TO_IDX[c] for c in seq])

        y = torch.zeros((3,MAX_LEN))

        if idx < self.pos_len:
            y[0] = make_vector(row.cdr3_start,row.cdr3_end)
            y[1] = make_vector(row.v_sequence_start,row.v_sequence_end)
            y[2] = make_vector(row.j_sequence_start,row.j_sequence_end)

        return X,y


def make_vector(start,end):
    v = torch.zeros(MAX_LEN)
    if pd.isna(start) or pd.isna(end):
        return v
    v[int(start)-1:int(end)] = 1
    return v


def dataset_to_pt(dataset, out_path):
    X = torch.stack([dataset[i][0] for i in range(len(dataset))])
    y = torch.stack([dataset[i][1] for i in range(len(dataset))])
    torch.save({'X': X, 'y': y}, out_path)