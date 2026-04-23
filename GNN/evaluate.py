import os
import torch
from torch_geometric.loader import DataLoader
from dataset import StructuralTimeDataset
from model import ST_GNN
from config import Config
from tqdm import tqdm

def evaluate():
    print(f"Loading dataset from {Config.DATA_DIR}...")
    dataset = StructuralTimeDataset(Config.DATA_DIR, seq_len=Config.SEQ_LEN)
    
    #Batch size can be larger during evaluation since we don't store gradients
    eval_batch_size = Config.BATCH_SIZE * 2
    loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
    
    print("Initializing ST-GNN...")
    model = ST_GNN().to(Config.DEVICE)
    
    #LOAD THE CHECKPOINT
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Error: Could not find {Config.MODEL_SAVE_PATH}.")
        print("You must let train.py finish at least one epoch to generate a save file!")
        return
        
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from Epoch {checkpoint['epoch']} (Training Loss: {checkpoint['loss']:.4f})\n")
    
    #SET TO EVALUATION MODE
    model.eval()
    
    total_disp_mae = 0.0
    total_force_mae = 0.0
    total_moment_mae = 0.0
    batch_count = 0
    
    print("Running Physical Accuracy Evaluation...")
    
    #Disable gradient calculation to save VRAM and speed up inference
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(Config.DEVICE)
            
            #Forward Pass
            pred_disp, pred_force, _ = model(batch)
            
            #REVERSE THE SCALING 
            real_pred_disp = pred_disp * Config.DISP_SCALE
            real_true_disp = batch.y_disp * Config.DISP_SCALE
            
            #Split and un-scale forces and moments
            real_pred_force = pred_force.clone()
            real_true_force = batch.y_force.clone()
            
            force_idx = [0, 1, 2, 6, 7, 8]
            moment_idx = [3, 4, 5, 9, 10, 11]
            
            real_pred_force[:, force_idx] *= Config.FORCE_SCALE
            real_pred_force[:, moment_idx] *= Config.MOMENT_SCALE
            real_true_force[:, force_idx] *= Config.FORCE_SCALE
            real_true_force[:, moment_idx] *= Config.MOMENT_SCALE
            
            #CALCULATE MEAN ABSOLUTE ERROR (MAE)
            disp_error = torch.abs(real_pred_disp - real_true_disp).mean().item()
            
            #Calculate Force and Moment errors separately so you can actually read them!
            force_error = torch.abs(real_pred_force[:, force_idx] - real_true_force[:, force_idx]).mean().item()
            moment_error = torch.abs(real_pred_force[:, moment_idx] - real_true_force[:, moment_idx]).mean().item()
            
            total_disp_mae += disp_error
            total_force_mae += force_error
            total_moment_mae += moment_error

            batch_count += 1
            
    #Calculate final averages
    avg_disp_mae = total_disp_mae / batch_count
    avg_force_mae = total_force_mae / batch_count
    avg_moment_mae = total_moment_mae / batch_count
    print(f"Average Displacement Error : {avg_disp_mae:.4f} mm")
    print(f"Average Internal Force Error : {avg_force_mae:.2f} Newtons")
    print(f"Average Moment Error: {avg_moment_mae:.2f} Nmm")

if __name__ == "__main__":
    evaluate()