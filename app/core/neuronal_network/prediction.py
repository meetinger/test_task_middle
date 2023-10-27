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

    model = Net()

    num_epochs = 32
    batch_size = 16
    learning_rate = 1e-2
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    # scheduler2 = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    full_data = UserActivityDataset(x, y)
    train_size = int(0.7 * len(full_data))
    val_size = len(full_data) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_data, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []

    # Обучение модели
    for epoch in range(num_epochs):

        train_loss = 0.0
        model.train()
        for data, labels in train_dataloader:

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        val_loss = 0.0
        model.eval()
        for data, labels in val_dataloader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * data.size(0)

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)


        print(f'Epoch {epoch+1} \n Train Loss: {train_loss / len(train_dataloader)} \n Val Loss: {val_loss / len(val_dataloader)}')
        scheduler1.step()
        # scheduler2.step()

    # Построение графика лосса
    plt.plot(list(range(0, num_epochs)), train_losses, label='Train loss')
    plt.plot(list(range(0, num_epochs)), val_losses, label='Valid loss')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.show()

    # Сравнение оригинала с предсказанием

    test_len = 200

    x_axis = []

    y_orig = []
    y_pred = []

    model.eval()
    with (torch.no_grad()):
        for i in range(test_len):
            data, label = full_data.x[i], full_data.y[i]
            output = model(data.unsqueeze(0))
            # print(output)

            # unscale_x = dataset_obj.x_scaler.inverse_transform(inputs.unsqueeze(0))
            # unscale_y_orig = dataset_obj.y_scaler.inverse_transform([[label.item()]])[0][0]
            # unscale_y_pred = dataset_obj.y_scaler.inverse_transform([[output.view(-1).item()]])[0][0]

            unscale_x = data.unsqueeze(0)
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
