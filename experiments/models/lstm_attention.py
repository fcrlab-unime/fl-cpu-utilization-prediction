import torch
import torch.nn as nn

class LSTMAttention(nn.Module):

    def __init__(self,
            input_size: int,
            hidden_size: int,
            num_classes: int,
            num_layers: int
        ):

        super(LSTMAttention, self).__init__()
        
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_classes = num_classes

        self.dropout1 = nn.Dropout(p=0.2)

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True
        )
        self.attention1 = nn.Linear(hidden_size*2, 1) # 2 is bidrectional

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        dropout1_out = self.dropout1(x)
        lstm1_out, _ = self.lstm1(dropout1_out)
        attention1_weights = torch.softmax(
            self.attention1(lstm1_out).squeeze(-1), dim=-1
        )
        context1_vector = torch.sum(
            lstm1_out * attention1_weights.unsqueeze(-1), dim=1
        )
        dropout2_out = self.dropout2(context1_vector)
        fc1_out = self.fc1(dropout2_out)
        return fc1_out