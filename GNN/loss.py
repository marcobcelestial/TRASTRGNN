import torch.nn as nn

class PhysicsLoss(nn.Module):
    def __init__(self, force_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.force_weight = force_weight # Balance displacement vs force loss magnitudes

    def forward(self, pred_disp, true_disp, pred_force, true_force):
        loss_disp = self.mse(pred_disp, true_disp)
        loss_force = self.mse(pred_force, true_force)
        
        return loss_disp + (self.force_weight * loss_force)