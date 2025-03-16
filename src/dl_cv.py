import os
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn
from dotenv import load_dotenv
import numpy as np
import wandb
from argparsing import get_model_from_args
from dataset import SpectralData

from torch_functions import test_best_model, train_test_with_early_stopping

if __name__ == "__main__":
    load_dotenv()
    config, model = get_model_from_args()
    
    data = SpectralData(
        domain_list = config.domain_list,
        trait_list = config.trait_list,
        directory_path = config.directory_path
    )
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data.input_other)):
        # Reset model
        model = get_model_from_args()[1] 
        
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
            # Dataloader
            train_dataset = TensorDataset(
                torch.tensor(data.input_other[train_idx], dtype = torch.float32),
                torch.tensor(data.target_other[train_idx], dtype = torch.float32),
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            validation_dataset = TensorDataset(
                torch.tensor(data.input_other[val_idx], dtype = torch.float32),
                torch.tensor(data.target_other[val_idx], dtype = torch.float32),
            )
            validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
            
            test_dataset = TensorDataset(
                torch.tensor(data.input_test, dtype = torch.float32),
                torch.tensor(data.target_test, dtype = torch.float32),
            )
            test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)
            
            optimizer = Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay) 
            loss_fn = nn.MSELoss()
            
            model = train_test_with_early_stopping(
                model, train_loader, validation_loader, test_loader, optimizer, loss_fn, epochs = config.epochs
            )
            
            optimizer = Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay) 
            r2_score = test_best_model(model, test_loader, optimizer, loss_fn)
            r2_scores.append(r2_score)

            wandb.finish()
        
    print(f"r2_score: {np.mean(r2_scores)}")