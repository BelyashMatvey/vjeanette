import time
import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vjeanette.parser import FastqDataset
from vjeanette.utils import select_best_strand, reverse_complement
from vjeanette.config import DEFAULT_THRESHOLDS


IDX_TO_BYTE = np.array(
    [b'A', b'C', b'G', b'T', b'N', b'P'],
    dtype='S1'
)


@torch.jit.script
def sequence_score_smoothed(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    window_size: int
) -> torch.Tensor:
    """
    Maximum average probability inside a valid window.

    logits:  [B, L]
    lengths: [B]

    return: [B]
    """

    probs = torch.sigmoid(logits).unsqueeze(1)

    smoothed = F.avg_pool1d(
        probs,
        kernel_size=window_size,
        stride=1
    ).squeeze(1)

    positions = torch.arange(
        smoothed.shape[1],
        device=logits.device
    ).unsqueeze(0)

    valid = (
        positions + window_size
        <= lengths.unsqueeze(1)
    )

    smoothed = smoothed.masked_fill(
        ~valid,
        -torch.inf
    )

    return smoothed.max(dim=1).values


class InferenceRunner:

    def __init__(
        self,
        model,
        device="cuda",
        thresholds=None,
        use_amp=True,
        compiled=False
    ):

        self.model = model
        self.device = torch.device(device)

        self.thresholds = (
            thresholds
            if thresholds is not None
            else DEFAULT_THRESHOLDS
        )

        self.use_amp = (
            use_amp
            and self.device.type == "cuda"
        )

        self.compiled = compiled

        self.logger = self._setup_logger()

        self.model = self.model.to(self.device)
        self.model.eval()

    def _setup_logger(self):

        logger = logging.getLogger("inference")
        logger.setLevel(logging.INFO)

        return logger

    def add_log_handler(self, log_file):

        handler = logging.FileHandler(log_file)

        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(message)s"
            )
        )

        self.logger.addHandler(handler)
        self.logger.propagate = False


    def _process_batch(
        self,
        x,
        read_ids
    ):
        """
        Process one batch.

        Returns:
            fasta_records
        """

        bs = len(read_ids)

        # ==================================================
        # 1. CPU -> GPU
        # ==================================================

        x = x.to(
            self.device,
            non_blocking=True
        )

        # Real sequence length before padding
        lengths = (
            x != 5
        ).sum(dim=1)

        # ==================================================
        # 2. Reverse complement on GPU
        # ==================================================

        x_rc = reverse_complement(x)

        x_both = torch.cat(
            (x, x_rc),
            dim=0
        )

        lengths_both = torch.cat(
            (lengths, lengths),
            dim=0
        )

        # ==================================================
        # 3. Model inference
        # ==================================================

        with torch.inference_mode():

            if self.use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    cdr3_logits, v_logits, j_logits = (
                        self.model(x_both)
                    )

            else:

                cdr3_logits, v_logits, j_logits = (
                    self.model(x_both)
                )

        # ==================================================
        # 4. Sequence-level scores
        # ==================================================

        # V
        v_score = sequence_score_smoothed(
            v_logits,
            lengths_both,
            window_size=15
        )

        # J
        j_score = sequence_score_smoothed(
            j_logits,
            lengths_both,
            window_size=10
        )

        # CDR3
        cdr_score = sequence_score_smoothed(
            cdr3_logits,
            lengths_both,
            window_size=6
        )

        # ==================================================
        # 5. Forward / reverse
        # ==================================================

        vf_score = v_score[:bs]
        vr_score = v_score[bs:]

        jf_score = j_score[:bs]
        jr_score = j_score[bs:]

        cdrf_score = cdr_score[:bs]
        cdrr_score = cdr_score[bs:]

        # ==================================================
        # 6. Existence predictions
        # ==================================================

        has_vf = (
            vf_score
            > self.thresholds["v_exist"]
        )

        has_vr = (
            vr_score
            > self.thresholds["v_exist"]
        )

        has_jf = (
            jf_score
            > self.thresholds["j_exist"]
        )

        has_jr = (
            jr_score
            > self.thresholds["j_exist"]
        )

        has_cdrf = (
            cdrf_score
            > self.thresholds["cdr_exist"]
        )

        has_cdrr = (
            cdrr_score
            > self.thresholds["cdr_exist"]
        )

        # ==================================================
        # 7. Select best strand
        # ==================================================

        choose_f = select_best_strand(
            vf_score,
            vr_score,
            jf_score,
            jr_score,
            has_vf,
            has_vr,
            has_jf,
            has_jr
        )

        f = choose_f.bool()

        # ==================================================
        # 8. Select predictions
        # ==================================================

        has_v = torch.where(
            f,
            has_vf,
            has_vr
        )

        has_j = torch.where(
            f,
            has_jf,
            has_jr
        )

        has_cdr3 = torch.where(
            f,
            has_cdrf,
            has_cdrr
        )

        # ==================================================
        # 9. SCORE
        #
        # Formula unchanged
        # ==================================================

        score = torch.where(

            f,

            2 * has_vf.float()
            + 2 * has_jf.float()
            + vf_score
            + jf_score,

            2 * has_vr.float()
            + 2 * has_jr.float()
            + vr_score
            + jr_score
        )

        # ==================================================
        # 10. Keep only positive reads
        # ==================================================

        keep = (
            has_v
            | has_j
            | has_cdr3
        )

        indices = torch.nonzero(
            keep,
            as_tuple=False
        ).flatten()

        if indices.numel() == 0:

            return []

        # ==================================================
        # 11. Move only retained reads to CPU
        # ==================================================

        indices_cpu = (
            indices
            .cpu()
            .numpy()
        )

        score_keep_cpu = (
            score[indices]
            .float()
            .cpu()
            .numpy()
        )

        v_keep_cpu = (
            has_v[indices]
            .cpu()
            .numpy()
        )

        j_keep_cpu = (
            has_j[indices]
            .cpu()
            .numpy()
        )

        cdr_keep_cpu = (
            has_cdr3[indices]
            .cpu()
            .numpy()
        )

        # ==================================================
        # 12. Select strand sequence
        # ==================================================

        selected_x = torch.where(
            f[indices, None],
            x[indices],
            x_rc[indices]
        )

        selected_x = (
            selected_x
            .cpu()
            .numpy()
        )

        # ==================================================
        # 13. FASTA output
        # ==================================================

        fasta_output = []

        for k, idx in enumerate(indices_cpu):

            sequence = (
                IDX_TO_BYTE[selected_x[k]]
                .tobytes()
                .rstrip(b'P')
                .decode("ascii")
            )

            header = (
                f">{read_ids[idx]} "
                f"score={score_keep_cpu[k]:.6f} "
                f"has_v={int(v_keep_cpu[k])} "
                f"has_j={int(j_keep_cpu[k])} "
                f"has_cdr3={int(cdr_keep_cpu[k])}"
            )

            fasta_output.append(
                header
                + "\n"
                + sequence
                + "\n"
            )

        return fasta_output


    def run(
        self,
        fastq_file,
        output_fasta,
        batch_size=32768,
        num_workers=1
    ):

        self.logger.info(
            f"PARAMETERS:\n"
            f"Fastq file: {fastq_file}\n"
            f"Output FASTA: {output_fasta}\n"
            f"Batch size: {batch_size}\n"
            f"Workers: {num_workers}\n"
            f"Device: {self.device}\n"
            f"AMP: {self.use_amp}\n"
            f"Compile: {self.compiled}\n"
        )

        dataset = FastqDataset(
            fastq_file,
            batch_size,
            num_workers
        )

        loader_kwargs = dict(

            dataset=dataset,

            batch_size=None,

            num_workers=num_workers,

            pin_memory=(
                self.device.type == "cuda"
            )
        )

        if num_workers > 0:

            loader_kwargs.update({

                "persistent_workers": True,

                "prefetch_factor": 2
            })

        loader = DataLoader(
            **loader_kwargs
        )

        start = time.perf_counter()

        total_reads = 0
        kept_reads = 0

        # ==================================================
        # FASTA OUTPUT
        # ==================================================

        with open(
            output_fasta,
            "w",
            buffering=1024 * 1024
        ) as fasta_file:

            for x, read_ids in loader:

                total_reads += len(
                    read_ids
                )

                fasta_records = (
                    self._process_batch(
                        x,
                        read_ids
                    )
                )

                if fasta_records:

                    fasta_file.writelines(
                        fasta_records
                    )

                    kept_reads += len(
                        fasta_records
                    )

        elapsed = (
            time.perf_counter()
            - start
        )

        speed = (
            total_reads
            / elapsed
        )

        self.logger.info(
            f"TIME: {elapsed:.2f} sec"
        )

        self.logger.info(
            f"READS: {total_reads}"
        )

        self.logger.info(
            f"KEPT: {kept_reads}"
        )

        self.logger.info(
            f"SPEED: {speed:.2f} reads/sec"
        )

        self.logger.info(
            f"OUTPUT FASTA: {output_fasta}"
        )