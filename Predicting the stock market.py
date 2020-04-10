import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
df = pd.read_csv('sphist.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.reset_index(drop=True)
df = df.reset_index()

df.rename({'index':'day'},inplace=True,axis=1)

# average price from the past X days
def average_price(row,days):
    if row['day'] < days:
        return 0
    else:
        day_start = row['day'] - days
        day_end = row['day'] - 1
        return np.mean(df.loc[day_start:day_end,'Close'])

df['mean_5_days'] = df.apply(average_price,axis=1,days=5)
df['mean_30_days'] = df.apply(average_price,axis=1,days=30)
df['mean_365_days'] = df.apply(average_price,axis=1,days=365)

df = df.iloc[365:]
df = df.dropna(axis=0)

train = df[df['Date'] < datetime(year=2013,month=1,day=1)]
test = df[df['Date'] >= datetime(year=2013,month=1,day=1)]

features = ['mean_5_days','mean_30_days','mean_365_days']
lr = LinearRegression()
lr.fit(train[features],train['Close'])
predictions = lr.predict(test[features])

mae = mean_absolute_error(predictions,test['Close'])

print(mae)