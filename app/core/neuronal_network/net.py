import torch
from torch import nn


class Net(nn.Module):
    """
    Модель нейронной сети

    X:
    1 - номер дня месяца
    2 - номер дня недели
    3 - номер месяца
    4 - номер года

    Y:
    1 - количество запросов в день

    """

    def __init__(self):
        super(Net, self).__init__()

        self.input_size = 4
        self.hidden_size = 16
        self.output_size = 1
        self.num_layers = 2

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_size, out_features=self.output_size),
        )


    def forward(self, x):
        x = x.unsqueeze(0)
        #
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        #
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        #
        # dropout_out = self.dropout(gru_out)
        #
        fc_out = self.fc(lstm_out)
        # #
        return fc_out.squeeze(0)
        # return self.layers(x)
