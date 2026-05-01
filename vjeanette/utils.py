import torch
from vjeanette.config import RC_TABLE

def reverse_complement(x):
    """Reverse complement"""
    return RC_TABLE.to(x.device)[x].flip(1)

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

def select_best_strand(vf, vr, jf, jr, has_vf, has_vr, has_jf, has_jr):
    """Select best strand (forward/reverse)"""
    max_vf = vf.max(1).values
    max_vr = vr.max(1).values
    max_jf = jf.max(1).values
    max_jr = jr.max(1).values
    
    score_f = calculate_scores(has_vf, has_jf, max_vf, max_jf)
    score_r = calculate_scores(has_vr, has_jr, max_vr, max_jr)
    
    return score_f >= score_r, max_vf, max_vr, max_jf, max_jr