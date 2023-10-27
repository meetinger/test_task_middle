import asyncio
import random
import datetime as dt
import calendar
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

import app.db.crud.users_crud as user_crud
from app.db.database import get_db_ctx
from app.db.models import User
from app.db.models.users_models import UserActivity


async def generate_activity(user: User, start_date: dt.date, end_date: dt.date, db: AsyncSession) -> None:

    requests = []

    cur_date = start_date
    while cur_date <= end_date:

        week_day_idx = calendar.weekday(year=cur_date.year, month=cur_date.month, day=cur_date.day)

        req_dt = dt.datetime(year=cur_date.year, month=cur_date.month, day=cur_date.day, hour=1)

        # if week_day_idx == cur_date.month % 7:
        if week_day_idx == 5:
            # формула активности пользователя в зависимости от месяца
            for i in range(0, random.randint(5, 10)):
                requests.append(UserActivity(user=user, request_time=req_dt+dt.timedelta(hours=i)))
        elif random.randint(1, 100) <= 10:
            # случайный шум
            requests.append(UserActivity(user=user, request_time=req_dt + dt.timedelta(hours=1)))

        cur_date += dt.timedelta(days=1)

    db.add_all(requests)
    await db.commit()

    print(requests)


async def main():
    async with get_db_ctx() as db:

        user_db = await user_crud.get_user(uuid.UUID('a1873206-c474-4582-a246-52bf45cec1a4'), db)

        await generate_activity(user_db, dt.datetime(year=2020, month=10, day=1),
                                dt.datetime(year=2023, month=12, day=1), db=db)


if __name__ == '__main__':
    asyncio.run(main())
