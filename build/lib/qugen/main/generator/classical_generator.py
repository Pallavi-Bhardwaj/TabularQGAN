
import torch.nn as nn

import torch.nn.functional as F

class Classical_Generator(nn.Module):
    def __init__(self, inputs_shape, num_hidden_layers, num_dimensions):
        super().__init__()
        self.inputs_shape = inputs_shape
        self.num_hidden_layers = num_hidden_layers
        self.num_dimensions =   num_dimensions

        self.layers = nn.ModuleList()
        # add Input layer
        self.layers.append(nn.Linear(inputs_shape, num_dimensions))
        self.layers.append(nn.LeakyReLU())
        # add hidden layers
        for i in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(num_dimensions, num_dimensions))
            self.layers.append(nn.LeakyReLU())
        # add output layer
        self.layers.append(nn.Linear(num_dimensions, inputs_shape))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x