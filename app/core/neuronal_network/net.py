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
        self.hidden_size = 32
        self.output_size = 1
        self.num_layers = 1

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.1),

            nn.Linear(128, 256),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.005),

            nn.Linear(256, 128),
            nn.LeakyReLU(),
            # nn.Dropout(p=0.1),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x = x.unsqueeze(0)
        #
        # h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        #
        # gru_out, _ = self.gru(x, h_0)
        #
        # dropout_out = self.dropout(gru_out)
        #
        # fc_out_out = self.fc_out(dropout_out)
        # #
        # return fc_out_out.squeeze(0)
        return self.layers(x)
