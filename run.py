import os
import csv
import mmap
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import logging
import sys
from model.model import UNet1D_Embed
import argparse

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MAX_LEN = 512

CHAR_MAP = np.full(256, 4, dtype=np.uint8)
CHAR_MAP[ord("A")] = 0
CHAR_MAP[ord("C")] = 1
CHAR_MAP[ord("G")] = 2
CHAR_MAP[ord("T")] = 3
CHAR_MAP[ord("N")] = 4
CHAR_MAP[ord("P")] = 5

RC_TABLE = torch.tensor([3,2,1,0,4,5], dtype=torch.uint8)

def filling_bytes(seq_bytes):
    seq = seq_bytes[:MAX_LEN]
    if len(seq) < MAX_LEN:
        seq += b"P" * (MAX_LEN - len(seq))
    return seq

def seq_to_indices_bytes(seq_bytes):
    arr = np.frombuffer(seq_bytes, dtype=np.uint8)
    return CHAR_MAP[arr]

class FastqDataset(IterableDataset): 

    def __init__(self, path, batch_size, num_workers):
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def parse_fastq(self, worker_id):

        f = open(self.path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        size = mm.size()
        chunk = size // self.num_workers

        start = worker_id * chunk
        end = size if worker_id == self.num_workers-1 else (worker_id+1)*chunk

        mm.seek(start)

        if start != 0:
            mm.readline()
            while True:
                pos = mm.tell()
                line = mm.readline()
                if not line:
                    break
                if line.startswith(b'@'):
                    mm.seek(pos)
                    break

        batch = []
        ids = []

        while mm.tell() < end:

            header = mm.readline()
            if not header:
                break

            seq = mm.readline().rstrip()
            mm.readline()
            mm.readline()

            read_id = header[1:].split()[0].decode()

            seq = filling_bytes(seq)

            batch.append(seq_to_indices_bytes(seq))
            ids.append(read_id)

            if len(batch) == self.batch_size:
                yield torch.from_numpy(np.stack(batch)).long(), ids
                batch, ids = [], []

        if batch:
            yield torch.from_numpy(np.stack(batch)).long(), ids

        mm.close()
        f.close()

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
        else:
            worker_id = worker_info.id

        yield from self.parse_fastq(worker_id)


def reverse_complement(x):
    return RC_TABLE.to(x.device)[x].flip(1)

def get_intervals_fast(probs, thr):

    mask = probs > thr

    any_mask = mask.any(1)
    mask_int = mask.to(torch.int8)
    starts = torch.argmax(mask_int, dim=1)

    ends = mask.size(1) - torch.argmax(mask_int.flip(1), dim=1) - 1

    starts[~any_mask] = -1
    ends[~any_mask] = -1

    return starts, ends

@torch.inference_mode()
def run_fastq_inference(model, fastq, out_csv, n_cores, device, logs,batch_size):
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(logs)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s")
    )

    logger.addHandler(handler)
    logger.propagate = False
    
    logger.info('Run inference')
    dataset = FastqDataset(fastq, batch_size, n_cores)

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=n_cores,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )


    thr_v_exist = 0.02
    thr_j_exist = 0.02
    thr_mask = 0.04

    model.eval()

    start = time.perf_counter()
    total = 0

    with open(out_csv, "w", newline="") as f:

        writer = csv.writer(f, delimiter="\t")

        writer.writerow([
            "read_id","strand",
            "has_v","has_j",
            "v_start","v_end",
            "j_start","j_end",
            "score"
        ])

        for x, read_ids in loader:
            bs = len(read_ids)
            total += bs

            x_rc = reverse_complement(x)

            x = torch.cat([x, x_rc], 0).to(device, non_blocking=True)

            with torch.autocast("cuda"):
                v_logits, j_logits = model(x)

            v_probs = torch.sigmoid(v_logits)
            j_probs = torch.sigmoid(j_logits)

            vf, vr = v_probs[:bs], v_probs[bs:]
            jf, jr = j_probs[:bs], j_probs[bs:]

            max_vf = vf.max(1).values
            max_vr = vr.max(1).values
            max_jf = jf.max(1).values
            max_jr = jr.max(1).values

            has_vf = max_vf > thr_v_exist
            has_vr = max_vr > thr_v_exist
            has_jf = max_jf > thr_j_exist
            has_jr = max_jr > thr_j_exist

            score_f = 2*has_vf.float() + 2*has_jf.float() + max_vf + max_jf
            score_r = 2*has_vr.float() + 2*has_jr.float() + max_vr + max_jr

            choose_f = score_f >= score_r

            vs_f, ve_f = get_intervals_fast(vf, thr_mask)
            js_f, je_f = get_intervals_fast(jf, thr_mask)

            vs_r, ve_r = get_intervals_fast(vr, thr_mask)
            js_r, je_r = get_intervals_fast(jr, thr_mask)

            vs_f[~has_vf] = -1
            ve_f[~has_vf] = -1
            js_f[~has_jf] = -1
            je_f[~has_jf] = -1

            vs_r[~has_vr] = -1
            ve_r[~has_vr] = -1
            js_r[~has_jr] = -1
            je_r[~has_jr] = -1

            rows = []

            for i, rid in enumerate(read_ids):

                if choose_f[i]:

                    rows.append([
                        rid,"forward",
                        int(has_vf[i]), int(has_jf[i]),
                        int(vs_f[i]), int(ve_f[i]),
                        int(js_f[i]), int(je_f[i]),
                        float(score_f[i])
                    ])

                else:

                    rows.append([
                        rid,"reverse",
                        int(has_vr[i]), int(has_jr[i]),
                        int(vs_r[i]), int(ve_r[i]),
                        int(js_r[i]), int(je_r[i]),
                        float(score_r[i])
                    ])

            writer.writerows(rows)

    dt=time.perf_counter()-start
    print("TIME:",dt)
    print("READS:",total)
    print("SPEED:",total/dt,"reads/sec")


def main():
    parser = argparse.ArgumentParser('Description')
    
    parser.add_argument('--in_file')
    
    parser.add_argument('--device', nargs='?',default='cuda')
    
    parser.add_argument('--n_cores', type=int,nargs='?', default = 1)
    
    parser.add_argument('--batch_size', type=int, nargs='?',default = 32768)
    
    parser.add_argument('--logs',nargs='?', default='./logs/progress.log')
    
    parser.add_argument('--out_file', nargs='?',default = f'./out/out.pt')
    
    args = parser.parse_args()
    model=UNet1D_Embed().to(args.device)
    model.load_state_dict(torch.load("./model/model_v1.pth",  weights_only = True))
    model=torch.compile(
        model,
        mode="reduce-overhead"
    )

    model=model.to(memory_format=torch.channels_last)
    run_fastq_inference(
        model = model,
        fastq = args.in_file,
        out_csv = args.out_file,
        n_cores = args.n_cores,
        device = args.device,
        logs = args.logs,
        batch_size = args.batch_size
    )

if __name__=="__main__":
    main()