import torch

# Constants
MAX_LEN = 512
RC_TABLE = torch.tensor([
    3,  # A -> T
    2,  # C -> G
    1,  # G -> C
    0,  # T -> A
    4,  # N -> N
    5   # P -> P
], dtype=torch.long)
# Deafults Thresholds
DEFAULT_THRESHOLDS = {
    'v_exist': 0.5,
    'j_exist': 0.5,
    'cdr_exist': 0.5
}

# CUDA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True