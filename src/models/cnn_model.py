
import torch.nn as nn
from torch.nn import functional as F

from models.base_model import BaseExperimentArgs, BaseExperimentModel
from torch_functions import init_weights

class CNNModelArgs(BaseExperimentArgs):
    input_dim: int = 2151
    output_dim: int = 1
    layer_number: int = 1
    kernel_size: int = 3
    padding: int = 1
    stride: int = 1
    
    
class CNNModel(BaseExperimentModel, nn.Module):
    def __init__(self, config: CNNModelArgs):
        nn.Module.__init__(self)
        BaseExperimentModel.__init__(self)
        self.name = "CNN"
        self.config = config
        
        self.conv1 = nn.Conv1d(
            in_channels = 1, 
            out_channels = 32, 
            kernel_size = config.kernel_size,
            padding = config.padding,
            stride = config.stride
        )
        
        self.conv2 = nn.Conv1d(
            in_channels = 32, 
            out_channels = 64, 
            kernel_size = config.kernel_size,
            padding = config.padding,
            stride = config.stride
        )
        
        # Bottleneck structure after CNN
        self.layers = nn.ModuleList()
        current_dim = 64 * config.input_dim
        
        # Create bottleneck structure
        for _ in range(self.config.layer_number):
            next_dim = int(current_dim /2)
            self.layers.append(nn.Linear(current_dim, next_dim))
            current_dim = next_dim
        
        self.output_layer = nn.Linear(current_dim, self.config.output_dim)
    
        # Custom weight initialization
        self.apply(init_weights)
        
    def forward(self, input):
        # Permute input for the CNN: from batch, channel, sequence to channel, batch, sequence
        input = input.unsqueeze(1)
        input = input.permute(1,0,2)
        
        input = F.relu(self.conv1(input))
        input = F.relu(self.conv2(input))
        
        # Permute back to normal: from channel, batch, sequence to batch, channel, sequence 
        input = input.permute(1,0,2)
        
        # Flatten the output:
        input = input.view(input.size(0), -1)
        
        for layer in self.layers:
            input = F.relu(layer(input))
        return self.output_layer(input)
        
    def get_args_model():
        return CNNModelArgs