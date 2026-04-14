import torch

# Constants
MAX_LEN = 512
RC_TABLE = torch.tensor([3,2,1,0,4,5], dtype=torch.uint8)

# Deafults Thresholds
DEFAULT_THRESHOLDS = {
    'v_exist': 0.02,
    'j_exist': 0.02,
    'mask': 0.04
}

# CUDA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True