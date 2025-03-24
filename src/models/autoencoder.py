
import torch.nn as nn
from torch.nn import functional as F

from models.base_model import BaseExperimentArgs, BaseExperimentModel
from torch_functions import init_weights

class AutoencoderModelArgs(BaseExperimentArgs):
    input_dim: int = 2151
    output_dim: int = 2
    layer_number: int = 1
    
class EncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        current_dim = self.config.input_dim
        
        # Create bottleneck structure
        for _ in range(self.config.layer_number):
            next_dim = int(current_dim /2)
            self.layers.append(nn.Linear(current_dim, next_dim))
            current_dim = next_dim
        
        self.output_layer = nn.Linear(current_dim, self.config.output_dim)
    
    def forward(self, input):
        for layer in self.layers:
            input = F.relu(layer(input))
        return self.output_layer(input)
        

class DecoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        current_dim = self.config.output_dim
        
        # Create bottleneck structure
        for _ in range(self.config.layer_number):
            next_dim = int(current_dim*2)
            self.layers.append(nn.Linear(current_dim, next_dim))
            current_dim = next_dim
        
        self.output_layer = nn.Linear(current_dim, self.config.input_dim)
    
    def forward(self, input):
        for layer in self.layers:
            input = F.relu(layer(input))
        return self.output_layer(input)

    
class AutoencoderModel(BaseExperimentModel, nn.Module):
    def __init__(self, config: AutoencoderModelArgs):
        nn.Module.__init__(self)
        BaseExperimentModel.__init__(self)
        self.name = "ACE"
        
        self.encoder = EncoderModel(config)
        self.decoder = DecoderModel(config)
    
        # Custom weight initialization
        self.apply(init_weights)
        
    def forward(self, input):
        input = self.encoder(input)
        input = self.decoder(input)
        return input
        
    def get_args_model():
        return AutoencoderModelArgs