#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta
import datetime
import scipy.stats as stats
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import datetime
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings


# In[ ]:


def dfa(x, n):
    """ Detrended Fluctuation Analysis (DFA) """
    # Calculate the cumulative sum of deviations from the mean
    y = np.cumsum(x - np.mean(x))
    # Create an array of sizes of windows to split the time series into
    ns = np.logspace(1, np.log10(n//2), 10).astype(int)
    # Calculate the fluctuations for each window size
    fluctuations = []
    for i in ns:
        # Split the time series into windows of size i
        windows = y[:n-(n%i)].reshape(-1, i)
        # Calculate the local trend for each window
        trends = np.array([np.polyfit(np.arange(i), w, 1)[0] for w in windows])
        # Calculate the fluctuations for each window
        fluctuations.append(np.sqrt(np.mean((windows - trends.reshape(-1,1)*np.arange(i))**2)))
    # Fit a straight line to the log-log plot of fluctuations vs. window size
    p = np.polyfit(np.log10(ns), np.log10(fluctuations), 1)
    # Return the slope of the line as the DFA exponent and the trend-cycle values
    return (y - np.polyval(p, np.log10(np.arange(1, n+1))))[n//2:]


# define the long-range dependence (LRD) function
def lrd_func(h, beta, gamma):
    return beta * np.abs(h)**gamma

# define the objective function to minimize
def objective(params, acf_values):
    beta, gamma = params
    residuals = acf_values - lrd_func(np.arange(1, len(acf_values)+1), beta, gamma)
    return np.sum(residuals**2)


# In[ ]:


data_1 = pd.read_csv("Input data.csv")

data_days = data_1.copy()
data_days_copy = data_1.copy()

data_1['date'] = pd.to_datetime(data_1['date'], format='%d-%m-%Y')
data_1 = data_1.groupby([pd.Grouper(key='date', freq='M')])['actual'].sum().reset_index(name ='actual')
data_1['date'] = data_1['date'].dt.to_period('M').dt.start_time
data_1['date'] = data_1['date'].dt.strftime('%Y-%m-%d')

data_2 = data_1.copy()
data_3 = data_1.copy()
data_1


# In[ ]:


data_11 = pd.read_csv("Input data.csv")

data_11['date'] = pd.to_datetime(data_11['date'], format='%d-%m-%Y')
data_11 = data_11.groupby([pd.Grouper(key='date', freq='1D')])['actual'].sum().reset_index(name ='actual')

data_11['date'] = data_11['date'].dt.strftime('%Y-%m-%d')
data_11


# In[ ]:


data_11 = pd.read_csv("Input data.csv")

data_11['date'] = pd.to_datetime(data_11['date'], format='%d-%m-%Y')
last_date = data_11['date'].max().date()
next_year_end = last_date.year
next_year_end = datetime.date(next_year_end +1, 12, 31)
days = (next_year_end - last_date).days
days


# In[ ]:


if data_1['actual'].count()>=4:
    if (data_1['actual'].astype(int) != 0).sum() >= 2:
        data_1.set_index('date', inplace=True)

        # convert the revenue data to a numpy array
        revenue = data_1['actual'].values.flatten()

        result = adfuller(revenue)
        
        if result[1] > 0.05:
            revenue_count = data_1['actual'].count()
   
            T = dfa(revenue, revenue_count)
  
            # repeat the array along a new axis to create a new shape (2, 12)
            T = np.tile(T, (2,))
            # reshape the array to have shape (24,)
            T = np.reshape(T, (revenue_count,))

        else:
        
            # calculate the autocorrelation function (ACF) of the revenue data
            revenue_acf = acf(revenue, nlags=len(revenue)-1)
            # estimate the beta and gamma parameters by minimizing the objective function
            initial_guess = [0.1, 0.5]
            result = minimize(objective, initial_guess, args=(revenue_acf,))
            beta_hat, gamma_hat = result.x

            # calculate the trend-cycle numerical values using the estimated beta and gamma parameters
            T = np.array([np.sum(revenue[i:] - np.mean(revenue)) / ((len(revenue)-i) * beta_hat * gamma_hat * (gamma_hat - 1)) for i in range(len(revenue))])
    
    
    
        if data_1['actual'].count()<24:
            # Decompose time series into trend, seasonality, cyclicality, and irregularity components
            result = sm.tsa.seasonal_decompose(data_1['actual'], model='additive', period=1)
        else:
            result = sm.tsa.seasonal_decompose(data_1['actual'], model='additive', period=12)
        
        # Extract the seasonal component from the decomposition
        S = np.array(result.seasonal)

        mean = np.mean(revenue)
        std = np.std(revenue)

        # Calculate the irregularity numerical value
        I = np.abs(revenue - mean) / std

        m1 = T+S+I
        m2 = (T+S)*I
        m3 = (T+I)*S
        m4 = (S+I)*T
        m5 = T*S+I
        m6 = T*I+S
        m7 = S*I+T
        m8 = T*S*I
        final_array = np.concatenate((m1,m2,m3,m4,m5,m6,m8))
        
        revenue_count = 48 - data_2['actual'].count()
        earliest_month = datetime.datetime.strptime(data_2['date'].min(), "%Y-%m-%d")

        months_list = []
        generated_months = set()

        while len(months_list) < revenue_count:
            earliest_month -= relativedelta(months=1)
            month = earliest_month.strftime("%Y-%m-%d")
        
            if month not in generated_months:
                generated_months.add(month)
                months_list.append(month)
  

        synthetic_data = {month: value for month, value in zip(months_list, final_array)}
        synthetic_data = pd.DataFrame({'date': list(synthetic_data.keys()), 'actual': list(synthetic_data.values())})
        synthetic_data['date'] = pd.to_datetime(synthetic_data['date'], infer_datetime_format=True)
        synthetic_data['date'] = synthetic_data['date'].dt.strftime('%Y-%m-%d')
        
        
        # Convert non-numeric values to NaN
        data_2['actual'] = data_2['actual'].apply(pd.to_numeric, errors='coerce')

        # Calculate mean and standard deviation of revenue data
        mean = data_2['actual'].mean()
        std_dev = data_2['actual'].std()

        # Set significance level and calculate critical value
        significance_level = 0.05  # Change this as needed
        critical_value = stats.norm.ppf(1 - significance_level / 2)

        # Calculate detection limit
        detection_limit = mean + critical_value * std_dev

        # Replace zero values with detection limit multiplied by 0.65
        data_2[data_2 == 0] = np.nan
        data = data_2.fillna(detection_limit * 0.65)


        df = pd.concat([synthetic_data, data])


        df['actual'] = df['actual'].apply(lambda x: abs(x))
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df.sort_values(by='date', ascending=True)



        # Convert date column to datetime format
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

        # Set date as index
        df.set_index('date', inplace=True)

        # Create a function to split monthly revenue to daily revenue
        def split_monthly_revenue(monthly_rev, num_days):
            daily_rev = np.random.gamma(shape=2, scale=monthly_rev/num_days/2, size=num_days)
            return daily_rev

        # Create a new dataframe to store the daily revenue
        days_df = pd.DataFrame(columns=['Date', 'actual'])

        # Iterate over the months and split monthly revenue to daily revenue
        for month, revenue in df.iterrows():
            # Get the number of days in the month
            num_days = month.days_in_month
        
            # Split monthly revenue to daily revenue
            daily_revenue = split_monthly_revenue(revenue['actual'], num_days)
        
            # Create a copy of the original array
            daily_revenue_copy = daily_revenue.copy()

            # Create a mask for non-zero elements
            mask = daily_revenue != 0

            # Apply the mask to the copy of the original array to get only non-zero values
            daily_revenue_copy = daily_revenue_copy[mask]

            # Take the natural logarithm of each non-zero element in the copy of the original array
            daily_revenue_copy = np.log(daily_revenue_copy)

            # Replace the non-zero elements in the original array with their logarithmic values
            daily_revenue[mask] = daily_revenue_copy


            # Add the daily revenue data to the dataframe
            days = pd.date_range(month, periods=num_days, freq='D')

            days_df = pd.concat([days_df, pd.DataFrame({'Date': days, 'actual': daily_revenue})], ignore_index=True)
            days_df = days_df.sort_values(by='Date', ascending=True)

        days_df = days_df.dropna()
        data = days_df.reset_index(drop=True)

        
        # Convert dates to datetime and set it as the index
        data['Date'] = pd.to_datetime(data['Date'])
        
                
        last_date = data['Date'].max()# get the maximum date from the 'date' column of the dataframe
        next_year_end = datetime.datetime(last_date.year+1, 1, 1) + timedelta(days=365) # get the end date of the next year
        days = (next_year_end - last_date).days # calculate the number of days between last date and next year end

        data.set_index('Date', inplace=True)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)

        # Split the data into training and testing sets
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # Define the function to create the dataset
        def create_dataset(dataset, look_back=1):
            X, Y = [], []
            for i in range(len(dataset)-look_back):
                X.append(dataset[i:(i+look_back), 0])
                Y.append(dataset[i+look_back, 0])
            return np.array(X), np.array(Y)

        # Create the training and testing datasets
        look_back = 7  # use the last 7 days to predict the next day's revenue
        X_train, Y_train = create_dataset(train_data, look_back)
        X_test, Y_test = create_dataset(test_data, look_back)

        # Reshape the input data for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Define the LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X_train, Y_train, epochs=2, batch_size=1, verbose=2, validation_data=(X_test, Y_test))
        
        # Predict the future 365 days
        last_look_back = scaled_data[-look_back:]  # use the last 7 days of available data to predict the next day's revenue
        future_predictions = []
        
        for i in range(days):
            x_input = np.reshape(last_look_back, (1, look_back, 1))
            prediction = model.predict(x_input, verbose=0)
            future_predictions.append(prediction[0])
            last_look_back = np.vstack((last_look_back[1:], prediction))

        # Inverse transform the predictions back to the original scale
        future_predictions = scaler.inverse_transform(future_predictions)

        values = np.exp(future_predictions)
        df = pd.DataFrame(values, columns=['predicted'])
        

        revenue_count = df['predicted'].count()

        dates_list = []
        generated_dates = set()

        while len(dates_list) < revenue_count:
            last_date += relativedelta(days=1)
            date = last_date.strftime("%Y-%m-%d")
        
            if date not in generated_dates:
                generated_dates.add(date)
                dates_list.append(date)


        df['Date'] = dates_list

        # Convert the date column to a datetime type
        df['Date'] = pd.to_datetime(df['Date'])

        #print(df)
        #print(df.to_dict('records'))
        
        # Group the DataFrame by month and sum the values
        monthly_df = df.groupby([pd.Grouper(key='Date', freq='M')])['predicted'].sum().reset_index(name ='predicted')
        monthly_df['Date'] = monthly_df['Date'].dt.to_period('M').dt.start_time
        monthly_df['Date'] = monthly_df['Date'].dt.strftime('%Y-%m-%d')

    elif (data_1['actual'].astype(int) != 0).sum() >= 1:
        
        revenue = data_11['actual'].values.flatten()
        mu = np.mean(revenue)
        sigma = np.std(revenue)
        final_array = norm.rvs(size = 100 - data_11['actual'].count(), loc=mu, scale=sigma)
        
        revenue_count = 100 - data_11['actual'].count()
        earliest_date = datetime.datetime.strptime(data_11['date'].min(), "%Y-%m-%d")

        dates_list = []
        generated_dates = set()

        while len(dates_list) < revenue_count:
            earliest_date -= relativedelta(days=1)
            date = earliest_date.strftime("%Y-%m-%d")
        
            if date not in generated_dates:
                generated_dates.add(date)
                dates_list.append(date)
        
        
        synthetic_data = {date: value for date, value in zip(dates_list, final_array)}
        synthetic_data = pd.DataFrame({'date': list(synthetic_data.keys()), 'actual': list(synthetic_data.values())})
        synthetic_data['date'] = pd.to_datetime(synthetic_data['date'], infer_datetime_format=True)
        synthetic_data['date'] = synthetic_data['date'].dt.strftime('%Y-%m-%d')


        df = pd.concat([synthetic_data, data_11])


        df['actual'] = df['actual'].apply(lambda x: abs(x))
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df.sort_values(by='date', ascending=True)
        
        last_date = df['date'].max()
        last_date = datetime.datetime.strptime(df['date'].max(), "%Y-%m-%d")# get the maximum date from the 'date' column of the dataframe
        next_year_end = datetime.datetime(last_date.year+1, 1, 1) + timedelta(days=365) # get the end date of the next year
        days = (next_year_end - last_date).days # calculate the number of days between last date and next year end
        
        df.set_index('date', inplace=True)

        # Fit ARIMA model
        model = ARIMA(df, order=(1,2,2))
        model_fit = model.fit()

        # Predict future values
        forecast = model_fit.forecast(steps=days)

        # Create index for the forecast period
        forecast_index = pd.date_range(start= df.index[-1], periods= len(forecast), freq='D')[1:]

        # Create dataframe with forecast values
        forecast_df = pd.DataFrame({'predicted': forecast}, index=forecast_index)

        # Add date column
        forecast_df = forecast_df.reset_index().rename(columns={'index': 'date'})
        
        forecast_df['predicted'] = forecast_df['predicted'].apply(lambda x: abs(x))
        forecast_df['date'] = pd.to_datetime(forecast_df['date'], format='%Y-%m-%d')
        forecast_df = forecast_df.sort_values(by='date', ascending=True)

        monthly_df = forecast_df.groupby([pd.Grouper(key='date', freq='M')])['predicted'].sum().reset_index(name ='predicted')
        monthly_df['date'] = monthly_df['date'].dt.to_period('M').dt.start_time
        monthly_df['date'] = monthly_df['date'].dt.strftime('%Y-%m-%d')
        
    else:
        print("Insufficient values to Predict")
        
elif data_1['actual'].count()<4:
    
    if (data_11['actual'].astype(int) != 0).sum() >= 1:
        
        revenue = data_11['actual'].values.flatten()
        mu = np.mean(revenue)
        sigma = np.std(revenue)
        final_array = norm.rvs(size = 100 - data_11['actual'].count(), loc=mu, scale=sigma)
        
        revenue_count = 100 - data_11['actual'].count()
        earliest_date = datetime.datetime.strptime(data_11['date'].min(), "%Y-%m-%d")

        dates_list = []
        generated_dates = set()

        while len(dates_list) < revenue_count:
            earliest_date -= relativedelta(days=1)
            date = earliest_date.strftime("%Y-%m-%d")
        
            if date not in generated_dates:
                generated_dates.add(date)
                dates_list.append(date)
        
        
        synthetic_data = {date: value for date, value in zip(dates_list, final_array)}
        synthetic_data = pd.DataFrame({'date': list(synthetic_data.keys()), 'actual': list(synthetic_data.values())})
        synthetic_data['date'] = pd.to_datetime(synthetic_data['date'], infer_datetime_format=True)
        synthetic_data['date'] = synthetic_data['date'].dt.strftime('%Y-%m-%d')


        df = pd.concat([synthetic_data, data_11])


        df['actual'] = df['actual'].apply(lambda x: abs(x))
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df.sort_values(by='date', ascending=True)
        
        last_date = df['date'].max()# get the maximum date from the 'date' column of the dataframe
        next_year_end = datetime.datetime(last_date.year+1, 1, 1) + timedelta(days=int(365)) # get the end date of the next year
        days = (next_year_end - last_date).days # calculate the number of days between last date and next year end
        
        df.set_index('date', inplace=True)

        # Fit ARIMA model
        model = ARIMA(df, order=(1,2,1))
        model_fit = model.fit()

        # Predict future values
        forecast = model_fit.forecast(steps=days)

        # Create index for the forecast period
        forecast_index = pd.date_range(start= df.index[-1], periods= len(forecast), freq='D')[1:]

        # Create dataframe with forecast values
        forecast_df = pd.DataFrame({'predicted': forecast}, index=forecast_index)

        # Add date column
        forecast_df = forecast_df.reset_index().rename(columns={'index': 'date'})
        
        forecast_df['predicted'] = forecast_df['predicted'].apply(lambda x: abs(x))
        forecast_df['date'] = pd.to_datetime(forecast_df['date'], format='%Y-%m-%d')
        forecast_df = forecast_df.sort_values(by='date', ascending=True)

        monthly_df = forecast_df.groupby([pd.Grouper(key='date', freq='M')])['predicted'].sum().reset_index(name ='predicted')
        monthly_df['date'] = monthly_df['date'].dt.to_period('M').dt.start_time
        monthly_df['date'] = monthly_df['date'].dt.strftime('%Y-%m-%d')
        
    else:
        print("Insufficient values to Predict")    
    


# In[ ]:


forecast_df


# In[ ]:


monthly_df


# In[ ]:


# Convert the date column to datetime format
monthly_df['date'] = pd.to_datetime(monthly_df['date'], format='%Y-%m-%d')

# Define a function to get the financial year based on the date
def get_financial_year(date):
    year = date.year
    if date.month < 4:
        year -= 1
    return str(year) + '-' + str(year+1)[2:]

# Apply the function to the date column and create a new column 'Financial Year'
monthly_df['Financial Year'] = monthly_df['date'].apply(get_financial_year)
monthly_df

