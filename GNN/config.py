import torch

class Config:
    #Paths
    DATA_DIR = r"D:\WU15\TRASTRGNN\Data\Version 7"
    MODEL_SAVE_PATH = "st_gnn_model.pth"
    
    #Model Architecture
    NODE_IN_CHANNELS = 6      #UX, UY, UZ, RX, RY, RZ
    EDGE_IN_CHANNELS = 4      #Area, Iz, Iy, J
    HIDDEN_DIM = 64
    FORCE_OUT_CHANNELS = 12   #6 DOFs per node per edge
    
    #Training
    BATCH_SIZE = 4            #Kept low for 6GB VRAM limits
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    SEQ_LEN = 3               #Number of past steps to look at

    DISP_SCALE = 100.0    
    FORCE_SCALE = 1e6     
    AREA_SCALE = 1e4
    INERTIA_SCALE = 1e9
    MOMENT_SCALE = 1e9    #1 Billion for N*mm

    LEARNING_RATE = 5e-4
    ACCUMULATION_STEPS = 4
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"