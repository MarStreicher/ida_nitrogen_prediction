from math import inf
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import torch
import wandb

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_epoch(model, dataloader, optimizer, loss_fn):
    # Set pytorch model in trainings mode
    model.train()
    losses = []
    r2_scores = []
    mse_scores = []
    
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
        
        mse = mean_squared_error(targets_np, logits_np)
        mse_scores.append(mse)
    
    return np.mean(r2_scores), np.mean(mse_scores), np.mean(losses)

def test_epoch(model, dataloader, loss_fn):
    # Set pytorch model in trainings mode
    model.eval()
    
    losses = []
    r2_scores = []
    mse_scores = []
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
            
            mse = mean_squared_error(targets_np, logits_np)
            mse_scores.append(mse)
        
        return np.mean(r2_scores), np.mean(mse_scores), np.mean(losses)
    
def train_with_early_stopping(model, train_loader, validation_loader, optimizer, loss_fn, patience = 10, epochs = 100):
    patience_counter = 0
    best_loss_val = np.inf
    
    for epoch in range(epochs):
        # train
        r2_score, mse_score, loss = train_epoch(model, train_loader, optimizer, loss_fn)
    
        # validation
        r2_score_val, mse_score_val, loss_val = test_epoch(model, validation_loader, loss_fn)
        
        if best_loss_val > loss_val:
            best_loss_val = loss_val
            patience_counter = 0
            best_r2_score_val = r2_score_val
            best_r2_score = r2_score
            best_mse_score_val = mse_score_val
            best_mse_score = mse_score
            best_model_state = model.state_dict() 
            torch.save(best_model_state, f"trained_models/{str(model.name).lower()}_model.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            wandb.log({"early_stopping_epoch": epoch})
            break
        
        wandb.log({
            "r2_train": best_r2_score,
            "mse_train": best_mse_score,
            "train_loss": loss,
            "r2_validation": best_r2_score_val,
            "mse_validation": best_mse_score_val,
            "validation_loss": loss_val,
        })
        
    return model
            
def test_best_model(model, test_loader, loss_fn):
    model.load_state_dict(torch.load(f"trained_models/{str(model.name).lower()}_model.pth", weights_only=True))
    model.eval()

    r2_score_test, mse_score_test, loss_test = test_epoch(model, test_loader, loss_fn)
    
    wandb.log({
        "r2_test": r2_score_test,
        "mse_test": mse_score_test,
        "test_loss": loss_test,
    })
    
    return r2_score_test, mse_score_test
           
    
            
