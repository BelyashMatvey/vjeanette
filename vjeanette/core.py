import time
import logging
import csv
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vjeanette.parser import FastqDataset
from vjeanette.utils import reverse_complement, select_best_strand
from vjeanette.config import DEFAULT_THRESHOLDS


IDX_TO_CHAR = np.array(['A', 'C', 'G', 'T', 'N', 'P'])


@torch.jit.script
def sequence_score_smoothed(
    logits: torch.Tensor,
    window_size: int
) -> torch.Tensor:
    """
    Максимальная средняя вероятность в непрерывном окне.

    logits: [B, L]
    return: [B]
    """
    probs = torch.sigmoid(logits).unsqueeze(1)

    smoothed = F.avg_pool1d(
        probs,
        kernel_size=window_size,
        stride=1
    ).squeeze(1)

    return smoothed.max(dim=1).values


class InferenceRunner:

    def __init__(
        self,
        model,
        device="cuda",
        thresholds=None,
        use_amp=True,
        compile_model=False
    ):
        self.model = model
        self.device = torch.device(device)
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.use_amp = use_amp and self.device.type == "cuda"

        self.logger = self._setup_logger()

        # Переносим модель на GPU один раз
        self.model = self.model.to(self.device)
        self.model.eval()

        # Опционально torch.compile
        if compile_model:
            self.logger.info("Compiling model with torch.compile()")
            self.model = torch.compile(
                self.model,
                mode="max-autotune"
            )

    def _setup_logger(self):
        logger = logging.getLogger("inference")
        logger.setLevel(logging.INFO)
        return logger

    def add_log_handler(self, log_file):
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s")
        )
        self.logger.addHandler(handler)
        self.logger.propagate = False

    @staticmethod
    def _tensor_to_sequence(x):
        """
        x: [L]
        Быстро переводит encoded sequence -> str.
        P (5) считается padding и обрезается.
        """
        arr = x.detach().cpu().numpy()

        # Убираем padding справа
        arr = arr[arr != 5]

        return ''.join(IDX_TO_CHAR[arr])

    def _process_batch(self, x, read_ids):
        """
        Обрабатывает batch и возвращает только FASTA-записи,
        прошедшие хотя бы один из V/J/CDR3 критериев.
        """

        bs = len(read_ids)

        # --------------------------------------------------
        # 1. Reverse complement
        # --------------------------------------------------

        x_rc = reverse_complement(x)

        # Один transfer CPU -> GPU вместо двух
        x_both = torch.cat((x, x_rc), dim=0).to(
            self.device,
            non_blocking=True
        )

        # --------------------------------------------------
        # 2. Model inference
        # --------------------------------------------------

        with torch.inference_mode():

            if self.use_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    cdr3_logits, v_logits, j_logits = self.model(
                        x_both
                    )
            else:
                cdr3_logits, v_logits, j_logits = self.model(
                    x_both
                )

        # --------------------------------------------------
        # 3. Разделяем forward / reverse
        # --------------------------------------------------

        vf = v_logits[:bs]
        vr = v_logits[bs:]

        jf = j_logits[:bs]
        jr = j_logits[bs:]

        cdrf = cdr3_logits[:bs]
        cdrr = cdr3_logits[bs:]

        # --------------------------------------------------
        # 4. Sequence-level scores
        # --------------------------------------------------

        vf_score = sequence_score_smoothed(vf, 15)
        vr_score = sequence_score_smoothed(vr, 15)

        jf_score = sequence_score_smoothed(jf, 10)
        jr_score = sequence_score_smoothed(jr, 10)

        cdrf_score = sequence_score_smoothed(cdrf, 6)
        cdrr_score = sequence_score_smoothed(cdrr, 6)

        # --------------------------------------------------
        # 5. Existence predictions
        # --------------------------------------------------

        has_vf = vf_score > self.thresholds["v_exist"]
        has_vr = vr_score > self.thresholds["v_exist"]

        has_jf = jf_score > self.thresholds["j_exist"]
        has_jr = jr_score > self.thresholds["j_exist"]

        has_cdrf = cdrf_score > self.thresholds["cdr_exist"]
        has_cdrr = cdrr_score > self.thresholds["cdr_exist"]

        # --------------------------------------------------
        # 6. Выбор strand
        # --------------------------------------------------

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

        # --------------------------------------------------
        # 7. Выбираем результаты для лучшей strand
        # --------------------------------------------------

        # Boolean mask по batch
        f = choose_f.bool()

        has_v = torch.where(
            f, has_vf, has_vr
        )

        has_j = torch.where(
            f, has_jf, has_jr
        )

        has_cdr3 = torch.where(
            f, has_cdrf, has_cdrr
        )

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

        # --------------------------------------------------
        # 8. Оставляем только V/J/CDR3 positive reads
        # --------------------------------------------------

        keep = has_v | has_j | has_cdr3

        indices = torch.nonzero(
            keep,
            as_tuple=False
        ).flatten()

        if indices.numel() == 0:
            return []

        # --------------------------------------------------
        # 9. Переносим маленький subset на CPU
        # --------------------------------------------------

        indices_cpu = indices.cpu().numpy()

        score_cpu = score[indices].float().cpu().numpy()
        v_cpu = has_v[indices].cpu().numpy()
        j_cpu = has_j[indices].cpu().numpy()
        cdr_cpu = has_cdr3[indices].cpu().numpy()

        choose_cpu = f[indices].cpu().numpy()

        # --------------------------------------------------
        # 10. FASTA
        # --------------------------------------------------

        output = []

        for k, idx in enumerate(indices_cpu):

            # Выбранная strand
            if choose_cpu[k]:
                seq_tensor = x[idx]
                strand = "forward"
            else:
                seq_tensor = x_rc[idx]
                strand = "reverse"

            sequence = self._tensor_to_sequence(seq_tensor)

            # Заголовок:
            #
            # >ID score=... has_v=... has_j=... has_cdr3=...
            #
            header = (
                f">{read_ids[idx]} "
                f"score={score_cpu[k]:.6f} "
                f"has_v={int(v_cpu[k])} "
                f"has_j={int(j_cpu[k])} "
                f"has_cdr3={int(cdr_cpu[k])}"
            )

            output.append(
                header + "\n" +
                sequence + "\n"
            )

        return output

    def run(
        self,
        fastq_file,
        output_fasta,
        batch_size=32768,
        num_workers=1,
        compiling=False
    ):

        self.logger.info(
            f"PARAMETERS:\n"
            f"Fastq file: {fastq_file}\n"
            f"Output FASTA: {output_fasta}\n"
            f"Batch size: {batch_size}\n"
            f"Workers: {num_workers}\n"
            f"Device: {self.device}\n"
            f"AMP: {self.use_amp}\n"
            f"Compile: {compiling}\n"
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
            pin_memory=(self.device.type == "cuda")
        )

        # Эти параметры нельзя нормально использовать
        # при num_workers=0
        if num_workers > 0:
            loader_kwargs.update({
                "persistent_workers": True,
                "prefetch_factor": 4
            })

        loader = DataLoader(**loader_kwargs)

        start = time.perf_counter()

        total_reads = 0
        kept_reads = 0

        with open(
            output_fasta,
            "w",
            buffering=1024 * 1024
        ) as fasta_file:

            for x, read_ids in loader:

                total_reads += len(read_ids)

                fasta_records = self._process_batch(
                    x,
                    read_ids
                )

                if fasta_records:
                    fasta_file.writelines(fasta_records)
                    kept_reads += len(fasta_records)

        elapsed = time.perf_counter() - start

        speed = total_reads / elapsed

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
            f"OUTPUT: {output_fasta}"
        )