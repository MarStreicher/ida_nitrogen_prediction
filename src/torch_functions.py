import numpy as np
from sklearn.base import r2_score
import torch
import wandb


def train_epoch(model, dataloader, optimizer, loss_fn):
    # Set pytorch model in trainings mode
    model.train()
    losses = []
    r2_scores = []
    
    for batch in dataloader:
        # Clear the gradients of all optimized tensors
        optimizer.zero_grad()
        
        input_features = batch[0]
        targets = batch[1]
        
        # 1. Forward pass
        logits = model(input_features)
        
        # 2. Compute error
        # Most of the loss functions nee the target tensor in a floating point format
        error = loss_fn(logits, targets.float())
        losses.append(error.item())
        
        # 3. Backpropagation
        error.backward()
        
        # 4. Parameter update
        optimizer.step()
        
        # 5. Accurary monitoring
        logits_np = logits.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy() 
        
        r2 = r2_score(targets_np, logits_np) 
        r2_scores.append(r2)

    wandb.log({
        "r2_train": np.mean(r2_scores),
        "train_loss": np.mean(losses)
    })

def test_epoch(model, dataloader, loss_fn):
    # Set pytorch model in trainings mode
    model.eval()
    
    losses = []
    r2_scores = []
    with torch.no_grad():
        for batch in dataloader:
            input_features = batch[0]
            targets = batch[1]
            
            # 1. Forward pass
            logits = model(input_features)
            
            # 2. Compute error
            error = loss_fn(logits, targets.float())
            losses.append(error.item())
            
            # 3. Accurary monitoring
            logits_np = logits.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy() 
            
            r2 = r2_score(targets_np, logits_np) 
            r2_scores.append(r2)

    wandb.log({
        "r2_train": np.mean(r2_scores),
        "train_loss": np.mean(losses)
    })
            
    #TODO: Normalization in the dataset class
    #TODO: Early stopping 
            
