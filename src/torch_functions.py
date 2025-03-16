from math import inf
import numpy as np
from sklearn.metrics import r2_score
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
        
        # 5. Accuracy monitoring
        logits_np = logits.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy() 
        
        r2 = r2_score(targets_np, logits_np) 
        r2_scores.append(r2)
    
    return np.mean(r2_scores), np.mean(losses)

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
            
            # 3. Accuracy monitoring
            logits_np = logits.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy() 
            
            r2 = r2_score(targets_np, logits_np) 
            r2_scores.append(r2)
        
        return np.mean(r2_scores), np.mean(losses)
    
def train_with_early_stopping(model, train_loader, validation_loader, optimizer, loss_fn, patience = 10, epochs = 100):
    patience_counter = 0
    best_loss_val = -np.inf
    best_model_weights = None
    
    for epoch in range(epochs):
        # train
        r2_score, loss = train_epoch(model, train_loader, optimizer, loss_fn)
    
        # validation
        r2_score_val, loss_val = test_epoch(model, validation_loader, loss_fn)
        
        if best_loss_val > loss_val:
            best_loss_val = loss_val
            patience_counter = 0
            best_model_weights = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            wandb.log({"early_stopping_epoch": epoch})
            break
        
        model.load_state_dict(best_model_weights)
        return r2_score, loss, r2_score_val, loss_val, model
            
            
            
            
    
            
