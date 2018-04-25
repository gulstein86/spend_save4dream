#import relevant package
import pandas as pd
from fbprophet import Prophet

#declare variable
today_spend = 200 # get the value from today spending

#import data from sample
df = pd.read_csv('myTrx.csv', thousands=',', parse_dates=['date'], skiprows=1 , names=['date','acc_name','description','category','amount_rm','amount'])

df['daysofweek'] = df['date'].dt.weekday
df['days'] = df['date'].dt.weekday_name
df['week'] = df['date'].dt.week

df_group = df.groupby(['week','daysofweek','date'],as_index=False)[['amount']].sum()
df_trx = df_group[['date','amount']][(df_group['week']>=20) & (df_group['week']<=21)]
df_trx.columns = ['ds','y']

#if needed please remove outlier first
df_trx.loc[(df_trx['ds'] == '2013-05-21'), 'y'] = None

#adding holidays into the algorithm
holiday = pd.DataFrame({
  'holiday': 'malaysia_holiday',
  'ds': pd.to_datetime(['2013-05-01', '2013-05-06', '2013-05-24', 
                        '2013-06-01', '2013-06-06']),
  'lower_window': 0,
  'upper_window': 1,
})


#starting to construct the algorithm
m = Prophet(weekly_seasonality=True, holidays=holiday)

m.fit(df_trx)
future = m.make_future_dataframe(periods=7) #extend the date to another 7 days

forecast = m.predict(future) #predict the future consumption


today_predict = forecast.set_index('ds', drop = False).loc['2013-05-31','yhat']
percent_used = today_spend / today_predict * 100

if percent_used <= 100:
    print('You have save {:.0f}% today.'.format(100 - percent_used))
else:
    print('OMG, You have exceed spending of {:.0f}% today.'.format(percent_used - 100))