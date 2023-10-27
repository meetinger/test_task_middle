import asyncio
import random
import uuid

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import datetime as dt

from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader

import app.db.crud.users_crud as user_crud
from app.core.neuronal_network.dataset import load_dataset_by_user, UserActivityDataset
from app.core.neuronal_network.net import Net
from app.db.database import get_db_ctx


async def predict(dataset: tuple, start_date: dt.date, end_date: dt.date):
    seed = 42

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    x, y = dataset

    dataset_obj = UserActivityDataset(x, y)

    model = Net()

    learning_rate = 1e-3
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # scheduler1 = ExponentialLR(optimizer, gamma=0.8)
    # scheduler2 = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    batch_size = 128
    dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False)

    num_epochs = 32

    train_losses = []
    test_losses = []

    # Обучение модели
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)  # Передача актуального значения hidden state
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # if i == 0:
            #     with torch.no_grad():
            #         test_inputs, test_labels = iter(dataloader).next()
            #         test_hidden = torch.zeros(model.num_layers, batch_size, model.hidden_size)
            #         test_outputs, _ = model(test_inputs, test_hidden)  # Передача актуального значения hidden state
            #         test_loss = criterion(test_outputs, test_labels)
            #         test_losses.append(test_loss.item())

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, '
              # f'Test Loss: {test_loss.item():.4f}'
              )
        # scheduler1.step()
        # scheduler2.step()

    # Построение графика лосса
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Testing loss')
    plt.legend(frameon=False)
    plt.show()

    # Сравнение оригинала с предсказанием

    test_len = 200

    x_axis = []

    y_orig = []
    y_pred = []

    with (torch.no_grad()):
        for i in range(test_len):
            inputs, label = dataset_obj.x[i], dataset_obj.y[i]
            output = model(inputs.unsqueeze(0))
            # print(output)

            # unscale_x = dataset_obj.x_scaler.inverse_transform(inputs.unsqueeze(0))
            # unscale_y_orig = dataset_obj.y_scaler.inverse_transform([[label.item()]])[0][0]
            # unscale_y_pred = dataset_obj.y_scaler.inverse_transform([[output.view(-1).item()]])[0][0]

            unscale_x = inputs.unsqueeze(0)
            unscale_y_orig = label.item()
            unscale_y_pred = output.view(-1).item()

            y_orig.append(unscale_y_orig)
            y_pred.append(unscale_y_pred)

            print(f"Original: {unscale_y_orig}, Predicted: {unscale_y_pred}")

    plt.plot(list(range(test_len)), y_orig, label='Orig')
    plt.plot(list(range(test_len)), y_pred, label='Pred')
    plt.legend()
    plt.show()


async def main():
    async with get_db_ctx() as db:
        user_db = await user_crud.get_user(uuid.UUID('a1873206-c474-4582-a246-52bf45cec1a4'), db)

        dataset = await load_dataset_by_user(user=user_db, db=db)

    await predict(dataset, start_date=dt.date(year=2023, month=2, day=24),
                  end_date=dt.date.today())


if __name__ == '__main__':
    asyncio.run(main())
