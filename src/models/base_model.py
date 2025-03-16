from abc import abstractmethod
import numpy as np
from pydantic import BaseModel
from typing import Optional, Type
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

class BaseExperimentArgs(BaseModel):
    model: str
    use_wandb: bool = True
    epochs: int = 2000
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    directory_path: str = "data"
    domain_list: list[str] = ["UNL_Maize", "UNL_Camelina", "UNL_Sorghum", "UNL_Soybean"]
    trait_list: list[str] = ["N"]

class BaseExperimentModel():
    def __init__(self):
        pass
    
    def predict(self, input_test):
        return self.model.predict(input_test)
    
    def evaluate(self, input_test, target_test):
        y_pred = self.predict(input_test)
        
        r2 = [r2_score(target_test[:, index], y_pred[:, index]) for index in range(target_test.shape[1])]
        r = [np.corrcoef(y_pred[:, index].flatten(), target_test[:, index].flatten()) for index in range(target_test.shape[1])]
        
        if target_test.shape[1] == 1:
            r = r[0][0][1]
        return r2, r
    
    @classmethod
    def get_args_model(cls) -> Type[BaseExperimentArgs]:
        raise NotImplementedError()
     