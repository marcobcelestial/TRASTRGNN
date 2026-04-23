import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from model import ST_GNN
from config import Config

def infer_and_visualize(pt_file, timestep_to_predict=5):
    print(f"Inferring step {timestep_to_predict} for {pt_file}...")
    
    #Load Data & Model
    raw_data = torch.load(pt_file, weights_only=False)
    model = ST_GNN().to(Config.DEVICE)
    
    #Extract the model state from the checkpoint dictionary
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    #Prepare Input Sequence
    t = timestep_to_predict - Config.SEQ_LEN
    x_seq = raw_data["y_disp"][t : t + Config.SEQ_LEN]
    x_flat = x_seq.transpose(0, 1).reshape(raw_data["node_pos"].shape[0], -1)
    
    #Apply the correct scales to the inputs before prediction
    x_flat_scaled = x_flat / Config.DISP_SCALE
    
    scaled_edge_attr = raw_data["edge_attr"].clone()
    scaled_edge_attr[:, 0] = scaled_edge_attr[:, 0] / Config.AREA_SCALE
    scaled_edge_attr[:, 1] = scaled_edge_attr[:, 1] / Config.INERTIA_SCALE
    scaled_edge_attr[:, 2] = scaled_edge_attr[:, 2] / Config.INERTIA_SCALE
    scaled_edge_attr[:, 3] = scaled_edge_attr[:, 3] / Config.INERTIA_SCALE
    
    data = Data(
        x=x_flat_scaled,
        edge_index=raw_data["edge_index"],
        edge_attr=scaled_edge_attr
    ).to(Config.DEVICE)
    
    #Predict
    with torch.no_grad():
        pred_disp, pred_force, _ = model(data)
    
    #Un-scale back to real physical units
    pred_disp = (pred_disp * Config.DISP_SCALE).cpu().numpy()
    pred_force = pred_force.cpu().numpy()
    
    #Split the Force and Moment un-scaling for accurate physics
    force_idx = [0, 1, 2, 6, 7, 8]
    moment_idx = [3, 4, 5, 9, 10, 11]
    pred_force[:, force_idx] *= Config.FORCE_SCALE
    pred_force[:, moment_idx] *= Config.MOMENT_SCALE
    
    #Visualization Setup
    nodes_original = raw_data["node_pos"].numpy()
    edge_index = raw_data["edge_index"].numpy()
    
    #Exaggerate displacement by 50x for visual clarity
    DISP_SCALE_VISUAL = 50.0 
    nodes_displaced = nodes_original + (pred_disp[:, :3] * DISP_SCALE_VISUAL)
    
    #Use Axial Force (1st component of the 12-DOF force tensor) for coloring
    axial_forces = pred_force[:, 0]
    norm_forces = (axial_forces - axial_forces.min()) / (axial_forces.max() - axial_forces.min() + 1e-8)
    cmap = plt.get_cmap('coolwarm') #Red = Tension/High Force, Blue = Compression
    
    #Plot 3D Graph
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Predicted State (Step {timestep_to_predict}) - 50x Amplified", fontsize=14)
    
    #Plot Original Ghost Skeleton
    for e in range(edge_index.shape[1]):
        i, j = edge_index[0, e], edge_index[1, e]
        ax.plot([nodes_original[i, 0], nodes_original[j, 0]],
                [nodes_original[i, 1], nodes_original[j, 1]],
                [nodes_original[i, 2], nodes_original[j, 2]], 
                color='gray', linestyle=':', alpha=0.3)
                
    #Plot Predicted Displaced Skeleton
    for e in range(edge_index.shape[1]):
        i, j = edge_index[0, e], edge_index[1, e]
        color = cmap(norm_forces[e])
        ax.plot([nodes_displaced[i, 0], nodes_displaced[j, 0]],
                [nodes_displaced[i, 1], nodes_displaced[j, 1]],
                [nodes_displaced[i, 2], nodes_displaced[j, 2]], 
                color=color, linewidth=2.5)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #Point this to a single generated .pt file
    sample_file = r"D:\WU15\TRASTRGNN\Data\Version 7\Structure-1005.pt"
    infer_and_visualize(sample_file)