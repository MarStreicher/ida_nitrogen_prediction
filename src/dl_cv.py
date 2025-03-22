import os
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from dotenv import load_dotenv
import numpy as np
import wandb
from argparsing import get_model_from_args
from dataset import SpectralData

from models.autoencoder import AutoencoderModel
from torch_functions import test_best_model, train_with_early_stopping

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
    mse_scores = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data.input_other)):
        # Reset model
        model = get_model_from_args()[1]
        
        # wandb
        if config.use_wandb:
            wandb.login(key=os.environ["WB_KEY"], relogin=True)
            
        experiment_name = f"dl_fold_{str(fold+1)}_{str(config.model).lower()}"
        wandb.init(
            project = "ida_nitrogen_prediction",
            entity = "marleen-streicher",
            config = config,
            name = experiment_name,
            save_code = True,
            mode = "online" if config.use_wandb else "disabled", 
        )

        with wandb.run:
            # GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create a W&B Table
            table = wandb.Table(columns=["Type", "Logs"])
            table.add_data("fold", str(fold))
            table.add_data("device",f"{str(device)}")
            table.add_data("cuda available",f"{str(torch.cuda.is_available())}")
            table.add_data("number of gpus",f"{str(torch.cuda.device_count())}")
            if device == "cuda":
                table.add_data("device name",f"{str(torch.cuda.get_device_name(0))}")
            wandb.log({"log_table": table})
            
            model = model.to(device)
            
            # Dataloader
            train_dataset = TensorDataset(
                torch.tensor(data.input_other[train_idx], dtype = torch.float32).to(device),
                torch.tensor(data.target_other[train_idx], dtype = torch.float32).to(device),
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            validation_dataset = TensorDataset(
                torch.tensor(data.input_other[val_idx], dtype = torch.float32).to(device),
                torch.tensor(data.target_other[val_idx], dtype = torch.float32).to(device),
            )
            validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
            
            test_dataset = TensorDataset(
                torch.tensor(data.input_test, dtype = torch.float32).to(device),
                torch.tensor(data.target_test, dtype = torch.float32).to(device),
            )
            test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)
            
            if isinstance(model, AutoencoderModel):
                train_dataset = TensorDataset(
                    torch.tensor(data.input, dtype = torch.float32).to(device),
                    torch.tensor(data.input, dtype = torch.float32).to(device),
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                
                validation_loader = train_loader
                test_loader = train_loader
                
            
            optimizer = optim.SGD(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay) 
            loss_fn = nn.MSELoss()
            
            # Train
            model = train_with_early_stopping(
                model, train_loader, validation_loader, optimizer, loss_fn, device, epochs = config.epochs
            )
            
            # Test
            r2_score, mse_score = test_best_model(model, test_loader, loss_fn, device)
            r2_scores.append(r2_score)
            mse_scores.append(mse_score)

            wandb.finish()
        
    print(f"r2_score: {np.mean(r2_scores)}")
    print(f"mse_score: {np.mean(mse_scores)}")