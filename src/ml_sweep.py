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
        directory_path = config.directory_path,
        normalization = config.normalised,
    )
    
    # wandb
    if config.use_wandb:
        wandb.login(key=os.environ["WB_KEY"], relogin=True)
        
    experiment_name = f"ml_sweep_{str(config.model).lower()}"
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
        y_pred = model.predict(
            input_test = data.input_validation
        )
        
        r2, r = model.evaluate(
            input_test = data.input_validation,
            target_test = data.target_validation
        )
        
        log_data = {
            "r2_validation" : np.mean(r2),
            "r_validation" : np.mean(r),
        }
        
        data = [[y_real.item(), y_pred.item()] for (y_real, y_pred) in zip(np.array(data.target_validation), np.array(y_pred))]
        table = wandb.Table(data = data, columns=["real y", "predicted y"])
        
        wandb.log({"real vs predicted y": wandb.plot.scatter(table, "real y", "predicted y", title="real vs predicted y")})
        wandb.log(log_data)
        wandb.finish()
        
        
    
    
    
