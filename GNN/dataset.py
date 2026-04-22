import os
import glob
import torch
from torch_geometric.data import Data, Dataset
from config import Config

class StructuralTimeDataset(Dataset):
    def __init__(self, data_dir, seq_len=3):
        super().__init__()
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.seq_len = seq_len
        self.samples = []
        self._build_index()

    def _build_index(self):
        #Create rolling windows for autoregressive training
        for file in self.files:
            data = torch.load(file, weights_only=False)
            num_steps = data["y_disp"].shape[0]
            for t in range(num_steps - self.seq_len):
                self.samples.append((file, t))

    def len(self):
        return len(self.samples)

    def get(self, idx):
        file, t = self.samples[idx]
        raw_data = torch.load(file, weights_only=False)
        
        # 1. Extract raw sequence and targets
        x_seq = raw_data["y_disp"][t : t + self.seq_len] 
        y_disp_target = raw_data["y_disp"][t + self.seq_len]
        y_force_target = raw_data["y_force"][t + self.seq_len]
        
        # 2. Flatten sequence into node features
        num_nodes = raw_data["node_pos"].shape[0]
        x_flat = x_seq.transpose(0, 1).reshape(num_nodes, -1)

        # 3. SCALE THE TARGETS AND INPUTS
        x_flat_scaled = x_flat / Config.DISP_SCALE
        y_disp_target_scaled = y_disp_target / Config.DISP_SCALE
        
        # Split the 12-DOF vector into Forces and Moments
        y_force_target_scaled = y_force_target.clone()
        force_idx = [0, 1, 2, 6, 7, 8]
        moment_idx = [3, 4, 5, 9, 10, 11]
        
        y_force_target_scaled[:, force_idx] = y_force_target_scaled[:, force_idx] / Config.FORCE_SCALE
        y_force_target_scaled[:, moment_idx] = y_force_target_scaled[:, moment_idx] / Config.MOMENT_SCALE

        # 4. SCALE THE GEOMETRY (Edge Attributes)
        scaled_edge_attr = raw_data["edge_attr"].clone()
        scaled_edge_attr[:, 0] = scaled_edge_attr[:, 0] / Config.AREA_SCALE
        scaled_edge_attr[:, 1] = scaled_edge_attr[:, 1] / Config.INERTIA_SCALE
        scaled_edge_attr[:, 2] = scaled_edge_attr[:, 2] / Config.INERTIA_SCALE
        scaled_edge_attr[:, 3] = scaled_edge_attr[:, 3] / Config.INERTIA_SCALE

        # 5. Return the strictly scaled Data object
        return Data(
            x=x_flat_scaled,
            edge_index=raw_data["edge_index"],
            edge_attr=scaled_edge_attr,
            pos=raw_data["node_pos"],
            y_disp=y_disp_target_scaled,
            y_force=y_force_target_scaled
        )