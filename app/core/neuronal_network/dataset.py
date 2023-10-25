import asyncio
import collections
import uuid
import datetime as dt

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

import app.db.crud.users_crud as user_crud
from app.db.database import get_db_ctx
from app.db.models import User


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

    # x = np.array([date.strftime('%Y-%m-%d') for date in x])
    # y = np.array(y)
    #
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
