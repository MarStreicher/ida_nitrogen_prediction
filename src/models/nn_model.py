
import torch.nn as nn
from torch.nn import functional as F

from models.base_model import BaseExperimentArgs, BaseExperimentModel
from torch_functions import init_weights

class SimpleNNModelArgs(BaseExperimentArgs):
    input_dim: int = 2151
    output_dim: int = 1
    layer_number: int = 1
    
class SimpleNNModel(BaseExperimentModel, nn.Module):
    def __init__(self, config: SimpleNNModelArgs):
        nn.Module.__init__(self)
        BaseExperimentModel.__init__(self)
        self.name = "NN"
        self.config = config
        self.layers = nn.ModuleList([
            nn.Linear(self.config.input_dim, self.config.input_dim) for _ in range(self.config.layer_number)
        ])
        self.output_layer = nn.Linear(self.config.input_dim, self.config.output_dim)
    
        # Custom weight initialization
        self.apply(init_weights)
        
    def forward(self, input):
        for layer in self.layers:
            input = F.relu(layer(input))
        return self.output_layer(input)
           
    def get_args_model():
        return SimpleNNModelArgs
