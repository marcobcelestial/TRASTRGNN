import os
import torch
from torch_geometric.loader import DataLoader
from dataset import StructuralTimeDataset
from model import ST_GNN
from loss import PhysicsLoss
from config import Config
from tqdm import tqdm

def train():
    dataset = StructuralTimeDataset(Config.DATA_DIR, seq_len=Config.SEQ_LEN)
    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    #1. INITIALIZE EVERYTHING FIRST
    model = ST_GNN().to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = PhysicsLoss().to(Config.DEVICE)
    
    start_epoch = 0
    
    #2. LOAD CHECKPOINT (If it exists)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print("\nFound existing checkpoint. Resuming training...")
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        
        print(f"Resuming from Epoch {start_epoch} with loss {checkpoint['loss']:.4f}\n")
    else:
        print("\nNo checkpoint found. Starting fresh.\n")
        
    print(f"Training on {Config.DEVICE} with {len(dataset)} sequence samples.")
    print(f"Effective Batch Size: {Config.BATCH_SIZE * Config.ACCUMULATION_STEPS}")
    
    model.train()
    #Note: range starts at start_epoch now, so it doesn't rewind to zero!
    for epoch in range(start_epoch, Config.EPOCHS):
        total_loss = 0
        optimizer.zero_grad() 
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")):
            batch = batch.to(Config.DEVICE)
            
            pred_disp, pred_force, _ = model(batch)
            
            loss = criterion(pred_disp, batch.y_disp, pred_force, batch.y_force)
            
            loss = loss / Config.ACCUMULATION_STEPS
            loss.backward()
            
            if ((batch_idx + 1) % Config.ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(train_loader)):
                #Keep the gradient clipping safety net on
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += (loss.item() * Config.ACCUMULATION_STEPS)
            
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.6f}")
        
        scheduler.step(epoch_loss)
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss
        }
        torch.save(checkpoint, Config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()