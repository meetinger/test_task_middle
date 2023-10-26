import asyncio
import operator
import uuid
from functools import reduce
from typing import Self
import datetime as dt
import torch
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

    Y:
    1 - количество секунд активности в сутках

    """

    def __init__(self, x: list[list[int]], y: list[list[int]]):
        """Инит метод"""

        self._x = torch.tensor(x, dtype=torch.float)
        self._y = torch.tensor(y, dtype=torch.float)


    def __getitem__(self, idx):
        """Метод, необходимый для синтаксиса obj[idx]"""
        res = self._x[idx], self._y[idx]
        # print(res)
        return res

    def __len__(self):
        """Метод, необходимый для синтаксиса len(obj)"""

        return len(self._x)

    @classmethod
    async def from_user(cls, user: User, db: AsyncSession, max_session_delta: dt.timedelta = dt.timedelta(hours=5)) -> Self:
        """Создать датасет по пользователю"""

        user_requests = await user_crud.get_user_activity(user, db=db)
        if not user_requests:
            raise ValueError('No activity')

        x = []
        y = []

        cur_date: dt.date = user_requests[0].request_time.date()
        cur_sessions = []

        for i in range(1, len(user_requests)):
            prev_request = user_requests[i-1]
            cur_request = user_requests[i]

            if cur_request.request_time.date() != cur_date:
                if cur_sessions:
                    x.append([cur_date.day, cur_date.weekday(), cur_date.month])
                    y.append([reduce(operator.add, cur_sessions).total_seconds()])
                cur_date = cur_request.request_time.date()
                cur_sessions = []

            cur_delta = cur_request.request_time - prev_request.request_time
            if cur_delta <= max_session_delta:
                cur_sessions.append(cur_delta)

        print(len(x))
        print(len(y))

        return cls(x, y)


async def main():
    async with get_db_ctx() as db:
        user_db = await user_crud.get_user(uuid.UUID('713d523a-4265-40ec-b42f-3a6474236892'), db)

        dataset = await UserActivityDataset.from_user(user=user_db, db=db)

        print(dataset)


if __name__ == '__main__':
    asyncio.run(main())

