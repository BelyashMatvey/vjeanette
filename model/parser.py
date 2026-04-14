import mmap
import torch
import numpy as np
from torch.utils.data import IterableDataset

CHAR_MAP = np.full(256, 4, dtype=np.uint8)
CHAR_MAP[ord("A")] = 0
CHAR_MAP[ord("C")] = 1
CHAR_MAP[ord("G")] = 2
CHAR_MAP[ord("T")] = 3
CHAR_MAP[ord("N")] = 4
CHAR_MAP[ord("P")] = 5

MAX_LEN = 512

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