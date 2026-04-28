import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=128, num_layers=2, 
                output_dim=1, bidirectional=False, dropout=0.3):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, x):
        
        B, G, T, C = x.size()  
        x = x.view(B, G*T, C)  
        
        lstm_out, _ = self.lstm(x)  
        final_hidden = lstm_out[:, -1, :] 

        return self.fc(final_hidden)

