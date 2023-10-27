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
        super().__init__()

        self.input_size = 4
        self.hidden_size = 64
        self.output_size = 1
        self.num_layers = 2  # Добавляем два слоя LSTM

        # self.fc_in = nn.Linear(self.input_size, self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.unsqueeze(0)

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # fc_in = self.fc_in(x)

        lstm_out, (hn, cn) = self.lstm(x, (h_0, c_0))

        out = self.relu(lstm_out)

        fc_out = self.fc_out(out)

        return fc_out.squeeze(0)
