import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self,
                 num_classes: int, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int
                ):
        """Define the LSTM model.

        Args:
            input_size (int): the number of input features for each time step.
            hidden_size (int): the number of hidden units in each LSTM layer.
            num_layers (int): the number of stacked LSTM layers.
            num_classes (int): the size of the output (e.g., the number of predicted values)
        """
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # self.fc_1 =  nn.Linear(hidden_size, 512) # fully connected 1
        self.fc =  nn.Linear(hidden_size, num_classes) # fully connected 1
        # self.fc = nn.Linear(512, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        # out = self.fc_1(out) # first Dense
        # out = self.relu(out) # relu
        out = self.fc(out) # Final Output
        return out