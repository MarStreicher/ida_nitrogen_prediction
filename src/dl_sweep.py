import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
from dotenv import load_dotenv
import numpy as np
import wandb
from argparsing import get_model_from_args
from dataset import SpectralData

from torch_functions import train_epoch, test_epoch, train_with_early_stopping

if __name__ == "__main__":
    load_dotenv()
    config, model = get_model_from_args()
    
    data = SpectralData(
        domain_list = config.domain_list,
        trait_list = config.trait_list,
        directory_path = config.directory_path
    )
    
    # wandb
    if config.use_wandb:
        wandb.login(key=os.environ["WB_KEY"], relogin=True)
        
    experiment_name = f"dl_sweep_{str(config.model).lower()}"
    wandb.init(
        project = "ida_nitrogen_prediction",
        entity = "marleen-streicher",
        config = config,
        name = experiment_name,
        save_code = True,
        mode = "online" if config.use_wandb else "disabled", 
    )
    
    with wandb.run:
        # Create dataloader
        train_dataset = TensorDataset(
            torch.tensor(data.input_train, dtype = torch.float32),
            torch.tensor(data.target_train, dtype = torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        validation_dataset = TensorDataset(
            torch.tensor(data.input_validation, dtype = torch.float32),
            torch.tensor(data.target_validation, dtype = torch.float32),
        )
        validation_loader = DataLoader(validation_dataset, batch_size = 32, shuffle = True)
        
        optimizer = Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay) 
        loss_fn = nn.MSELoss()
        for epoch in range(config.epochs):
            r2_score, loss, r2_score_val, loss_val, _ = train_with_early_stopping(model, train_loader, validation_loader, optimizer, loss_fn, epochs = config.epochs)
            
            wandb.log({
                "r2_train": r2_score,
                "train_loss": loss,
                "r2_validation": r2_score_val,
                "validation_loss": loss_val,
            })
    
        wandb.finish()
        