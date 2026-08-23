import torch
from vjeanette.config import RC_TABLE
import torch.nn.functional as F
def reverse_complement(x):
    """
    Reverse complement with right-padding preservation.

    P = 5

    Example:
        ACGTACGPPPP
        ->
        CGTACGTPPPP
    """

    P = 5

    # Sequence length without padding
    lengths = (x != P).sum(dim=1)

    L = x.shape[1]

    # Reverse complement
    x_rc = RC_TABLE.to(x.device)[x].flip(1)

    positions = torch.arange(
        L,
        device=x.device
    ).unsqueeze(0)
    src_positions = (
        L - lengths.unsqueeze(1) + positions
    ).clamp(0, L - 1)

    rc_seq = torch.gather(
        x_rc,
        dim=1,
        index=src_positions
    )

    result = torch.full_like(x, P)

    mask = positions < lengths.unsqueeze(1)

    result[mask] = rc_seq[mask]

    return result
def get_intervals_fast(probs, thr):
    """Fast intervals defining"""
    mask = probs > thr
    
    any_mask = mask.any(1)
    mask_int = mask.to(torch.int8)
    starts = torch.argmax(mask_int, dim=1)
    ends = mask.size(1) - torch.argmax(mask_int.flip(1), dim=1) - 1
    
    starts[~any_mask] = -1
    ends[~any_mask] = -1
    
    return starts, ends

def calculate_scores(has_v, has_j, max_v, max_j):
    """Score calculation"""
    return 2*has_v.float() + 2*has_j.float() + max_v + max_j

def select_best_strand(vf_score, vr_score, jf_score, jr_score, has_vf, has_vr, has_jf, has_jr):
    """Select best strand (forward/reverse)"""
    score_f = (2 * has_vf.float()+ 2 * has_jf.float() + vf_score+ jf_score)

    score_r = (2 * has_vr.float()+ 2 * has_jr.float()+ vr_score+ jr_score)

    return score_f >= score_r