
import torch.nn as nn
from torch.nn import functional as F

from models.base_model import BaseExperimentArgs, BaseExperimentModel
from torch_functions import init_weights

class CNNModelArgs(BaseExperimentArgs):
    input_dim: int = 2151
    output_dim: int = 1
    layer_number: int = 1
    kernel_size: int = 3
    padding: int = 0
    stride: int = 2
    out_channels_conv1: int = 16
    out_channels_conv2: int = 32
    pooling: bool = True
    
class CNNModel(BaseExperimentModel, nn.Module):
    def __init__(self, config: CNNModelArgs):
        nn.Module.__init__(self)
        BaseExperimentModel.__init__(self)
        self.name = "CNN"
        self.config = config
        
        self.conv1 = nn.Conv1d(
            in_channels = 1, 
            out_channels = config.out_channels_conv1, 
            kernel_size = config.kernel_size,
            padding = config.padding,
            stride = config.stride
        )
        
        self.conv2 = nn.Conv1d(
            in_channels = config.out_channels_conv1, 
            out_channels = config.out_channels_conv2, 
            kernel_size = config.kernel_size,
            padding = config.padding,
            stride = config.stride
        )
        
        # Bottleneck structure after CNN
        self.layers = nn.ModuleList()
        output_size_conv1 = self._get_output_size(config.input_dim, config.kernel_size, config.padding, config.stride)
        output_size_conv2 = self._get_output_size(output_size_conv1, config.kernel_size, config.padding, config.stride)
        
        if self.config.pooling:
            output_size_pool = self._get_output_size(output_size_conv2, config.kernel_size, config.padding, config.stride)
            current_dim = config.out_channels_conv2 * output_size_pool
        else:
            current_dim = config.out_channels_conv2 * output_size_conv2
        
        if self.config.pooling:
            self.pool = nn.MaxPool1d(
                kernel_size = self.config.kernel_size,
                stride = self.config.stride,
                )
        
        # Create bottleneck structure
        for _ in range(self.config.layer_number):
            next_dim = int(current_dim /2)
            self.layers.append(nn.Linear(current_dim, next_dim))
            current_dim = next_dim
        
        self.output_layer = nn.Linear(current_dim, self.config.output_dim)
    
        # Custom weight initialization
        self.apply(init_weights)
        
    def forward(self, input):
        input = input.unsqueeze(1)
        
        input = F.relu(self.conv1(input))
        input = F.relu(self.conv2(input))
        
        if self.config.pooling:
            input = self.pool(input)
        
        # Flatten the output:
        input = input.view(input.size(0), -1)
        
        for layer in self.layers:
            input = F.relu(layer(input))
        return self.output_layer(input)
    
        
    def _get_output_size(self, input_size, kernel_size, padding, stride):
        return int(((input_size - kernel_size + 2 * padding)/stride)+1)
        
    def get_args_model():
        return CNNModelArgs