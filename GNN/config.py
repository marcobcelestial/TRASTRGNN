import torch

# Paths
DATA_DIR      = r"D:\WU15\TRASTRGNN\Data\Version 3"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "runs"

# Dataset
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
# remainder is test

# Graph construction
# Node features built from pos [N,3] + vm at t=0 [N,1] + fixed-base flag [N,1]
NODE_IN_DIM  = 5
# Edge features: relative displacement vector [3] + euclidean distance [1]
EDGE_IN_DIM  = 4

# Targets
# y_disp  [T, N, 3]   → node-level, 3 outputs per node
# y_vm    [T, N, 1]   → node-level, 1 output per node
# y_stress[T, C, 6]   → edge/element-level, 6 outputs per element
NODE_OUT_DIM = 4   # UX UY UZ + von Mises
ELEM_OUT_DIM = 6   # Sxx Syy Szz Sxy Syz Sxz

HIDDEN_DIM      = 128
NUM_GNN_LAYERS  = 6
NUM_GNN_HEADS   = 4    # for GAT layers
DROPOUT         = 0.1
# Temporal: GRU unrolled over T timesteps
GRU_HIDDEN_DIM  = 256
GRU_LAYERS      = 2

BATCH_SIZE    = 4       # graphs per batch
EPOCHS        = 200
LR            = 1e-3
LR_STEP       = 50      # reduce LR every N epochs
LR_GAMMA      = 0.5
WEIGHT_DECAY  = 1e-5
GRAD_CLIP     = 1.0     # max gradient norm

# Loss weights — disp is primary, vm and stress are auxiliary
LOSS_W_DISP   = 1.0
LOSS_W_VM     = 0.5
LOSS_W_STRESS = 0.3

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4     # DataLoader workers

SEED = 42