import torch
from torch import nn


class Net(nn.Module):
    """
    Модель нейронной сети
    Была использована LSTM, т.к она хорошо подходит для временных рядов

    Вход:
    1 - номер дня месяца
    2 - номер дня недели
    3 - номер месяца

    Выход:
    1 - количество секунд активности в сутках
    """

    def __init__(self):
        super().__init__()
        self.input_size = 3
        self.hidden_size = 128
        self.num_layers = 2
        self.out_size = 1

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)

        out = self.fc(lstm_out[:, -1, :])

        return out
