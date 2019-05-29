import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_dim, num_hidden, hidden_layers, num_classes, batch_size, device="cpu"):
        super(LSTM, self).__init__()
        self.device = device
        self.num_hidden = num_hidden
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=num_hidden, num_layers=hidden_layers, dropout=0).to(device)
        self.linear = nn.Linear(in_features=num_hidden, out_features=num_classes)

    def forward(self, input):
        lstm_output, _ = self.lstm(input)
        linear_output = self.linear(lstm_output[-1])
        return linear_output
