import csv
import time
import torch
import logging
from torch.utils.data import DataLoader
from model.parser import FastqDataset
from .utils import reverse_complement, get_intervals_fast, select_best_strand
from .config import DEFAULT_THRESHOLDS

class InferenceRunner:
    def __init__(self, model, device='cuda', thresholds=None):
        self.model = model
        self.device = device
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger("inference")
        logger.setLevel(logging.INFO)
        return logger
    
    def add_log_handler(self, log_file):
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self.logger.addHandler(handler)
        self.logger.propagate = False
    
    def _process_batch(self, x, read_ids):
        """Process one batch"""
        bs = len(read_ids)
        x_rc = reverse_complement(x)
        x = torch.cat([x, x_rc], 0).to(self.device, non_blocking=True)
        
        with torch.autocast("cuda"):
            v_logits, j_logits = self.model(x)
        
        v_probs = torch.sigmoid(v_logits)
        j_probs = torch.sigmoid(j_logits)
        
        vf, vr = v_probs[:bs], v_probs[bs:]
        jf, jr = j_probs[:bs], j_probs[bs:]
        
        has_vf = vf.max(1).values > self.thresholds['v_exist']
        has_vr = vr.max(1).values > self.thresholds['v_exist']
        has_jf = jf.max(1).values > self.thresholds['j_exist']
        has_jr = jr.max(1).values > self.thresholds['j_exist']
        
        choose_f, max_vf, max_vr, max_jf, max_jr = select_best_strand(
            vf, vr, jf, jr, has_vf, has_vr, has_jf, has_jr
        )
        
        # Получение интервалов
        vs_f, ve_f = get_intervals_fast(vf, self.thresholds['mask'])
        js_f, je_f = get_intervals_fast(jf, self.thresholds['mask'])
        vs_r, ve_r = get_intervals_fast(vr, self.thresholds['mask'])
        js_r, je_r = get_intervals_fast(jr, self.thresholds['mask'])
        
        # Применение масок
        self._apply_masks(vs_f, ve_f, has_vf)
        self._apply_masks(js_f, je_f, has_jf)
        self._apply_masks(vs_r, ve_r, has_vr)
        self._apply_masks(js_r, je_r, has_jr)
        
        return self._prepare_results(read_ids, choose_f, 
                                     vs_f, ve_f, js_f, je_f,
                                     vs_r, ve_r, js_r, je_r,
                                     has_vf, has_jf, has_vr, has_jr,
                                     choose_f, max_vf, max_vr, max_jf, max_jr)
    
    def _apply_masks(self, starts, ends, has_region):
        starts[~has_region] = -1
        ends[~has_region] = -1
    
    def _prepare_results(self, read_ids, choose_f, 
                        vs_f, ve_f, js_f, je_f,
                        vs_r, ve_r, js_r, je_r,
                        has_vf, has_jf, has_vr, has_jr,
                        choose_f_mask, max_vf, max_vr, max_jf, max_jr):
        
        rows = []
        for i, rid in enumerate(read_ids):
            if choose_f[i]:
                score = 2*has_vf[i] + 2*has_jf[i] + max_vf[i] + max_jf[i]
                rows.append([
                    rid, "forward",
                    int(has_vf[i]), int(has_jf[i]),
                    int(vs_f[i]), int(ve_f[i]),
                    int(js_f[i]), int(je_f[i]),
                    float(score)
                ])
            else:
                score = 2*has_vr[i] + 2*has_jr[i] + max_vr[i] + max_jr[i]
                rows.append([
                    rid, "reverse",
                    int(has_vr[i]), int(has_jr[i]),
                    int(vs_r[i]), int(ve_r[i]),
                    int(js_r[i]), int(je_r[i]),
                    float(score)
                ])
        return rows
    
    def run(self, fastq_file, output_csv, batch_size=32768, num_workers=1):
        """Main inference start method"""
        self.logger.info('Run inference')
        
        dataset = FastqDataset(fastq_file, batch_size, num_workers)
        loader = DataLoader(
            dataset, batch_size=None, num_workers=num_workers,
            pin_memory=True, persistent_workers=True, prefetch_factor=4
        )
        
        self.model.eval()
        start = time.perf_counter()
        total_reads = 0
        
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([
                "read_id", "strand", "has_v", "has_j",
                "v_start", "v_end", "j_start", "j_end", "score"
            ])
            
            for x, read_ids in loader:
                total_reads += len(read_ids)
                rows = self._process_batch(x, read_ids)
                writer.writerows(rows)
        
        elapsed = time.perf_counter() - start
        self.logger.info(f"TIME: {elapsed:.2f}")
        self.logger.info(f"READS: {total_reads}")
        self.logger.info(f"SPEED: {total_reads/elapsed:.2f} reads/sec")