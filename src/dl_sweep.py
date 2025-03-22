import os
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from dotenv import load_dotenv
import numpy as np
import wandb
from argparsing import get_model_from_args
from dataset import SpectralData
import pandas as pd

from models.autoencoder import AutoencoderModel
from torch_functions import train_with_early_stopping

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    with wandb.run:
        # Create a W&B Table
        table = wandb.Table(columns=["Type", "Logs"])
        table.add_data("device",f"{str(device)}")
        table.add_data("cuda available",f"{str(torch.cuda.is_available())}")
        table.add_data("number of gpus",f"{str(torch.cuda.device_count())}")
        if device == "cuda":
            table.add_data("device name",f"{str(torch.cuda.get_device_name(0))}")
        wandb.log({"log_table": table})
        
        # Create dataloader
        train_dataset = TensorDataset(
            torch.tensor(data.input_train, dtype = torch.float32).to(device),
            torch.tensor(data.target_train, dtype = torch.float32).to(device),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        validation_dataset = TensorDataset(
            torch.tensor(data.input_validation, dtype = torch.float32).to(device),
            torch.tensor(data.target_validation, dtype = torch.float32).to(device),
        )
        validation_loader = DataLoader(validation_dataset, batch_size = 32, shuffle = True)
        
        if isinstance(model, AutoencoderModel):
            train_dataset = TensorDataset(
                torch.tensor(data.input, dtype = torch.float32).to(device),
                torch.tensor(data.input, dtype = torch.float32).to(device),
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            validation_dataset = TensorDataset(
                torch.tensor(data.input, dtype = torch.float32).to(device),
                torch.tensor(data.input, dtype = torch.float32).to(device),
            )
            validation_loader = DataLoader(validation_dataset, batch_size = 32, shuffle = True)
        
        optimizer = optim.SGD(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay) 
        loss_fn = nn.MSELoss()

        _ = train_with_early_stopping(model, train_loader, validation_loader, optimizer, loss_fn, device, epochs = config.epochs)
    
        wandb.finish()
        