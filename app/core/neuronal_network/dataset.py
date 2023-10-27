import asyncio
import collections
import uuid
import datetime as dt

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sqlalchemy.ext.asyncio import AsyncSession
from torch.utils.data import Dataset

import app.db.crud.users_crud as user_crud
from app.db.database import get_db_ctx
from app.db.models import User

class UserActivityDataset(Dataset):
    """Датасет активности пользователя

    X:
    1 - номер дня месяца
    2 - номер дня недели
    3 - номер месяца
    4 - номер года

    Y:
    1 - количество запросов в день

    """

    def __init__(self, x: list[list[int]], y: list[list[int]]):
        """Инит метод"""

        self.x_scaler = MinMaxScaler(feature_range=(0,1))
        self.y_scaler = MinMaxScaler(feature_range=(0,1))

        # x = self.x_scaler.fit_transform(x)
        # y = self.y_scaler.fit_transform(y)

        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __getitem__(self, idx):
        """Метод, необходимый для синтаксиса obj[idx]"""
        res = self.x[idx], self.y[idx]
        return res

    def __len__(self):
        """Метод, необходимый для синтаксиса len(obj)"""
        return len(self.x)

    # def inverse_transform(self, x_normalized, y_normalized):
    #     x_original = self.x_scaler.inverse_transform(x_normalized)
    #     y_original = self.y_scaler.inverse_transform(y_normalized)
    #     return x_original, y_original


async def load_dataset_by_user(user: User, db: AsyncSession):
    requests = await user_crud.get_user_activity(user, db=db)

    counter = collections.defaultdict(lambda: 0)

    for req in requests:
        counter[req.request_time.date()] += 1

    x_min = min(counter.keys())
    x_max = max(counter.keys())

    cur_date = x_min

    while cur_date < x_max:
        if cur_date not in counter:
            counter[cur_date] = 0
        cur_date += dt.timedelta(days=1)

    x, y = zip(*sorted(counter.items()))

    x = [[d.day, d.isocalendar()[1], d.month, d.year] for d in x]
    # x = [[d.day, d.month, d.year] for d in x]
    # x = [[d.day, d.isocalendar()[1], d.month] for d in x]
    y = [[r] for r in y]

    return x, y


async def main():
    async with get_db_ctx() as db:
        user_db = await user_crud.get_user(uuid.UUID('713d523a-4265-40ec-b42f-3a6474236892'), db)

        x, y = await load_dataset_by_user(user=user_db, db=db)

        print(x)
        print(y)

        print(len(x))
        print(len(y))


if __name__ == '__main__':
    asyncio.run(main())
