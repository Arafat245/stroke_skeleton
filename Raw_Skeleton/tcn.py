import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, 
                              padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                              padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):

    def __init__(self, N_gaits, input_dim=96, num_channels=[16], kernel_size=3, 
                 dropout=0.3, output_dim=1):
        super(TCN, self).__init__()
        
        self.seq_len = 100*N_gaits
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size, 
                                   padding=padding, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.num_channels[-1] * self.seq_len, self.output_dim)

    def forward(self, x):

        B, G, T, C = x.shape
        
        x = x.view(B, G * T, C)  

        x = x.permute(0, 2, 1)  
        tcn_out = self.network(x)
        tcn_out = tcn_out.permute(0, 2, 1)  

        output = self.act(tcn_out)  
        output = self.dropout(output)
        
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output) 
        
        return output

