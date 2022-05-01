# build the model

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Creates a temporal block.
    Args:
        n_inputs (int): number of inputs.
        n_outputs (int): size of fully connected layers.
        kernel_size (int): kernel size along temporal axis of convolution layers within the temporal block.
        dilation (int): dilation of convolution layers along temporal axis within the temporal block.
        padding (int): padding
        dropout (float): dropout rate
    Returns:
        tuple of output layers
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        if padding == 0:
            self.net = nn.Sequential(
                weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)),
                nn.ReLU(), 
                nn.Dropout(dropout), 
                weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)), 
                nn.ReLU(), 
                nn.Dropout(dropout))
        else:
            self.net = nn.Sequential(
                weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)), 
                Chomp1d(padding), 
                nn.ReLU(), 
                nn.Dropout(dropout), 
                weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)), 
                Chomp1d(padding), 
                nn.ReLU(), 
                nn.Dropout(dropout))

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        return out, self.relu(out)



class Generator(nn.Module):
    # Causal temporal convolutional network with skip connections
    def __init__(self):
        super(Generator, self).__init__()
        self.tcn = nn.ModuleList([
            TemporalBlock(3, 80, kernel_size=1, stride=1, dilation=1, padding=0),
            TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=1, padding=1),
            TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=2, padding=2),
            TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=4, padding=4),
            TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=8, padding=8),
            TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=16, padding=16),
            TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=32, padding=32)])
        self.last = nn.Conv1d(80, 1, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return x


class Discriminator(nn.Module):
    # Causal temporal convolutional network with skip connections
    def __init__(self, seq_len, conv_dropout=0.05):
        super(Discriminator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(1, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                  TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=1, padding=1),
                                  TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=2, padding=2),
                                  TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=4, padding=4),
                                  TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=8, padding=8),
                                  TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=16, padding=16),
                                  TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=32, padding=32)])
        self.last = nn.Conv1d(80, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(127, 1), nn.Sigmoid())

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
#         print(x.shape)
        return self.to_prob(x).squeeze()t(np.nansum(a ** 2, axis=0) * np.nansum(b ** 2, axis=1))