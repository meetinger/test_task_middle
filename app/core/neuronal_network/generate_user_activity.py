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


async def generate_activity(user: User, start: dt.date, end: dt.date, day_average: dt.timedelta, db: AsyncSession) -> None:
    def generate_week(min_week_day: int, max_week_day: int, week_sum_activity: dt.timedelta):
        average = week_sum_activity.total_seconds() / 7
        days = [average for _ in range(7)]

        min_max_delta = random.uniform(0.05, 0.1) * average

        days[min_week_day] -= min_max_delta
        days[max_week_day] += min_max_delta

        other_delta = random.uniform(0.025, 0.05) * average

        idx = random.sample([i for i in range(7) if i not in (max_week_day, min_week_day)], k=2)

        days[idx[0]] += other_delta
        days[idx[1]] -= other_delta

        return days

    def generate_month(year: int, month: int, min_week_day: int,
                       max_week_day: int, month_day_average: dt.timedelta):
        days = []

        month_matrix = calendar.monthcalendar(year=year, month=month)

        for week in month_matrix:
            generated_week = generate_week(min_week_day, max_week_day, week_sum_activity=month_day_average * 7)
            for idx, day in enumerate(week):
                if day == 0:
                    continue
                days.append(generated_week[idx])

        return days

    requests = []

    cur_month_date = start.replace(day=1)

    while cur_month_date < end:
        week_days = generate_month(cur_month_date.year, cur_month_date.month, min_week_day=cur_month_date.month % 7,
                                   max_week_day=7 - cur_month_date.month % 7, month_day_average=day_average)

        for idx, day_time in enumerate(week_days, start=1):
            day_timedelta = dt.timedelta(seconds=day_time)
            request_dt = dt.datetime(year=cur_month_date.year, month=cur_month_date.month, day=idx, hour=1)
            day_requests = [UserActivity(user=user, request_time=request_dt),
                            UserActivity(user=user, request_time=request_dt + day_timedelta)]
            requests.extend(day_requests)

        cur_month_date = cur_month_date + dt.timedelta(
            days=calendar.monthrange(cur_month_date.year, cur_month_date.month)[1])

    db.add_all(requests)
    await db.commit()

    print(requests)


async def main():
    async with get_db_ctx() as db:

        user_db = await user_crud.get_user(uuid.UUID('713d523a-4265-40ec-b42f-3a6474236892'), db)

        await generate_activity(user_db, dt.datetime(year=2023, month=10, day=1), dt.datetime(year=2023, month=12, day=1),
                                dt.timedelta(hours=2), db=db)


if __name__ == '__main__':
    asyncio.run(main())
