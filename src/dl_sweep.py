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

from torch_functions import train_epoch, test_epoch

if __name__ == "__main__":
    load_dotenv()
    config, model = get_model_from_args()
    
    data = SpectralData(
        domain_list = config.domain_list,
        trait_list = config.trait_list,
        directory_path = config.directory_path
    )
    
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
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
    
    optimizer = Adam(model.parameters(), lr=config.learning_rate) 
    loss_fn = nn.MSELoss()
    for epoch in config.epochs:
        # train
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, loss_fn)
    
        # validation
        validation_loss, validation_accuracy = test_epoch(model, validation_loader, loss_fn)
    