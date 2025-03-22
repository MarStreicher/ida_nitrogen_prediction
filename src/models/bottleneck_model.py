
import torch.nn as nn
from torch.nn import functional as F

from models.base_model import BaseExperimentArgs, BaseExperimentModel
from torch_functions import init_weights

class BottleneckNNModelArgs(BaseExperimentArgs):
    input_dim: int = 2151
    output_dim: int = 1
    layer_number: int = 1
    drop_out: bool = False
    
class BottleneckNNModel(BaseExperimentModel, nn.Module):
    def __init__(self, config: BottleneckNNModelArgs):
        nn.Module.__init__(self)
        BaseExperimentModel.__init__(self)
        self.name = "NN"
        self.config = config
        
        self.layers = nn.ModuleList()
        current_dim = self.config.input_dim
        
        # Create bottleneck structure
        for _ in range(self.config.layer_number):
            next_dim = int(current_dim /2)
            self.layers.append(nn.Linear(current_dim, next_dim))
            current_dim = next_dim
        
        if config.drop_out:
            self.drop_out = nn.Dropout(p = 0.5)
        
        self.output_layer = nn.Linear(current_dim, self.config.output_dim)
    
        # Custom weight initialization
        self.apply(init_weights)
        
    def forward(self, input):
        for layer in self.layers:
            input = F.relu(layer(input))
        if hasattr(self, 'drop_out'):
            input = self.drop_out(input)
        return self.output_layer(input)
        
    def get_args_model():
        return BottleneckNNModelArgs
