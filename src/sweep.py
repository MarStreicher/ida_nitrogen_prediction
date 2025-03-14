import os
from dotenv import load_dotenv
import numpy as np
import wandb
from argparsing import get_model_from_args
from dataset import SpectralData

if __name__ == "__main__":
    load_dotenv()
    config, model = get_model_from_args()
    
    # data
    data = SpectralData(
        domain_list = config.domain_list,
        trait_list = config.trait_list,
        directory_path = config.directory_path
    )
    
    # wandb
    if config.use_wandb:
        wandb.login(key=os.environ["WB_KEY"], relogin=True)
        
    experiment_name = f"sweep_{str(config.model).lower()}"
    wandb.init(
        project = "ida_nitrogen_prediction",
        entity = "marleen-streicher",
        config = config,
        name = experiment_name,
        save_code = True,
        mode = "online" if config.use_wandb else "disabled", 
    )
    
    # sweep
    with wandb.run:
        # train
        model.train(
            input_train = data.input_train,
            target_train = data.target_train
        )
        
        # evaluate
        r2, r = model.evaluate(
            input_test = data.input_validation,
            target_test = data.target_validation
        )
        
        log_data = {
            "r2_validation" : np.mean(r2),
            "r_validation" : np.mean(r),
        }
        
        wandb.log(log_data)
        
        
    
    
    
