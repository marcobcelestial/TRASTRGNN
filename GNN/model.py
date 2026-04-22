import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from config import Config

class ST_GNN(nn.Module):
    def __init__(self):
        super().__init__()
        #Input size: 6 DOFs * SEQ_LEN
        in_dim = Config.NODE_IN_CHANNELS * Config.SEQ_LEN
        
        #Spatial Graph Convolution
        self.conv1 = SAGEConv(in_dim, Config.HIDDEN_DIM)
        self.conv2 = SAGEConv(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        
        #Temporal Memory (Node-wise GRU)
        self.gru = nn.GRUCell(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        
        #Node Decoder (Predicts next displacement)
        self.disp_decoder = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, Config.NODE_IN_CHANNELS)
        )
        
        #Edge Decoder (Predicts internal forces based on connected nodes and beam properties)
        self.force_decoder = nn.Sequential(
            nn.Linear((Config.HIDDEN_DIM * 2) + Config.EDGE_IN_CHANNELS, 64),
            nn.ReLU(),
            nn.Linear(64, Config.FORCE_OUT_CHANNELS)
        )

    def forward(self, data, h=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        #Spatial Message Passing
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        
        #Temporal Update (Memory)
        if h is None:
            h = torch.zeros(x.size(0), Config.HIDDEN_DIM, device=x.device)
        h = self.gru(x, h)
        
        #Predict Displacements
        pred_disp = self.disp_decoder(h)
        
        #Predict Forces (Concatenate node i, node j, and edge attributes)
        row, col = edge_index
        edge_features = torch.cat([h[row], h[col], edge_attr], dim=-1)
        pred_force = self.force_decoder(edge_features)
        
        return pred_disp, pred_force, h