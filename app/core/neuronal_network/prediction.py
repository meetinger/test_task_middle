import asyncio
import uuid

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import datetime as dt

import app.db.crud.users_crud as user_crud
from app.core.neuronal_network.dataset import load_dataset_by_user
from app.db.database import get_db_ctx


async def predict(dataset: tuple, start_date: dt.date, end_date: dt.date):
    # x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.2, shuffle=False,
    #                                                     random_state=0)

    x, y = dataset
    #x datetime.date list
    #y int list

    # x = np.array(x)
    # y = np.array(y)
    df = pd.DataFrame({'y': y}, index=x)

    df = df.asfreq('D')

    # model = SARIMAX(y, order=(1, 1, 0), seasonal_order=(1, 1, 0, 30), trend='n', dates=x)

    df = df.head(100)

    model = ARIMA(df, order=(1, 1, 0),  trend='n')
    model_fit = model.fit()

    test_range = list(range(len(x)))

    start, end = 100, 150

    forecast = model_fit.predict(start=start+1, end=end)

    plt.plot(test_range[start:end], forecast, label='predicted',
             color='red')

    plt.plot(test_range[start:end], y[start:end], label='real', color='blue')
    plt.legend(loc='best')
    plt.title('Predicted vs Real')
    plt.show()


async def main():
    async with get_db_ctx() as db:
        user_db = await user_crud.get_user(uuid.UUID('713d523a-4265-40ec-b42f-3a6474236892'), db)

        dataset = await load_dataset_by_user(user=user_db, db=db)

    await predict(dataset, start_date=dt.date(year=2023, month=2, day=24),
                  end_date=dt.date.today())

if __name__ == '__main__':
    asyncio.run(main())
