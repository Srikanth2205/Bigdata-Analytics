#!/usr/bin/env python
# coding: utf-8


# Code for SARIMA without the exogenous features.
# forecasts only based on PM2.5 from the given dataset.
# Auto Arima and (PACF and ACF) are used to verify the correct p,q,,d,P,D,Q 
# Values and the values are fixed to train SARIMAX

# In[1]:


import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import pmdarima as pm
import seaborn as sns
from sklearn.metrics import mean_squared_error


# In[2]:


sites= np.array(["PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
        "PRSA_Data_20130301-20170228/PRSA_Data_Changping_20130301-20170228.csv",
        "PRSA_Data_20130301-20170228/PRSA_Data_Dingling_20130301-20170228.csv",
        "PRSA_Data_20130301-20170228/PRSA_Data_Dongsi_20130301-20170228.csv",
        "PRSA_Data_20130301-20170228/PRSA_Data_Guanyuan_20130301-20170228.csv",
        "PRSA_Data_20130301-20170228/PRSA_Data_Gucheng_20130301-20170228.csv",
        "PRSA_Data_20130301-20170228/PRSA_Data_Huairou_20130301-20170228.csv",
        "PRSA_Data_20130301-20170228/PRSA_Data_Nongzhanguan_20130301-20170228.csv",
        "PRSA_Data_20130301-20170228/PRSA_Data_Shunyi_20130301-20170228.csv",
        "PRSA_Data_20130301-20170228/PRSA_Data_Tiantan_20130301-20170228.csv",
        "PRSA_Data_20130301-20170228/PRSA_Data_Wanliu_20130301-20170228.csv",
        "PRSA_Data_20130301-20170228/PRSA_Data_Wanshouxigong_20130301-20170228.csv"])




# In[4]:


print(sites.shape)

location = np.empty(12, dtype=object)


# In[5]:


for x in range(12):
    location[x] = pd.read_csv(sites[x])
    print(location[x].shape)
    print(location[x].isna().sum())

# In[6]:

for x in range(12):
    location[x]['PM2.5']=location[x]['PM2.5'].interpolate(method="linear")
    location[x]['SO2']=location[x]['SO2'].interpolate(method="linear")
    location[x]['NO2']=location[x]['NO2'].interpolate(method="linear")
    location[x]['CO']=location[x]['CO'].interpolate(method="linear")
    location[x]['O3']=location[x]['O3'].interpolate(method="linear")
    location[x]['TEMP']=location[x]['TEMP'].interpolate(method="linear")
    location[x]['PRES']=location[x]['PRES'].interpolate(method="linear")
    location[x]['DEWP']=location[x]['DEWP'].interpolate(method="linear")
    location[x]['RAIN']=location[x]['RAIN'].interpolate(method="linear")
    location[x]['WSPM']=location[x]['WSPM'].interpolate(method="linear")
    
for x in range(12):
    print(location[x].head(5))
    print(location[x].isna().sum())
    

# convert to datetime index
for x in range(12):
    location[x].index = pd.to_datetime(location[x][['year', 'month', 'day', 'hour']])
    location[x].drop(['No','year', 'month', 'day', 'hour', 'PM10', 'station','wd'], axis=1, inplace=True)
    print(location[x].index[0])
    print(location[x].index[-1])


for x in range(12):
    print(location[x].columns)
    

# In[7]:


for x in range(12):
    location[x] = location[x].rename({'PM2.5': 'PM2_5'}, axis=1)  # new method
    # descriptive stats
    print(location[x].PM2_5.describe())
    print("Null values :\n",location[x].isna().sum())
    print("\n\n")




# In[24]:


for x in range(12):
    # convert frequency to weeks
        print("modeling for ",sites[x],"is starting \n\n")
        location_wk = location[x].resample('W').mean()
        
        print('week frequency shape:', location_wk.shape)
        
        

        plt.figure(figsize=(10, 5))
        plt.plot(location[x].PM2_5)
        plt.title('Beijing Air Quality: PM2.5 Concentration from 2013 - 2017', fontsize=18)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('PM2.5 Concentration (ug / m^3)', fontsize=14);
        plt.style.use('seaborn-dark-palette')
        plt.rcParams['figure.figsize'] = (12, 6)

        decomposition = sm.tsa.seasonal_decompose(location_wk.PM2_5, model='additive', extrapolate_trend='freq')
        fig = decomposition.plot()
        location_aft = adfuller(location_wk.PM2_5, autolag='AIC')
        output = pd.Series(location_aft[0:4], index=['test statistic', 'pvalue', 'number of lags used', 'number of observations'])
        print(output)
        # train-test split
        pct_train = 0.80
        location_wk_index = round(len(location_wk) * pct_train)
        train_set, test_set = location_wk[:location_wk_index], location_wk[location_wk_index:]
        # ACF
        lag_acf = acf(train_set.PM2_5, nlags=20)

        plt.figure(figsize=(12, 10))
        plt.subplot(211)
        plt.stem(lag_acf)
        plt.axhline(y=0, linestyle='-', color='black')
        plt.axhline(y=-1.96/np.sqrt(len(train_set)), linestyle='--', color='gray')
        plt.axhline(y=1.96/np.sqrt(len(train_set)), linestyle='--', color='gray')
        plt.title('ACF Plot: PM2.5 Concentration', fontsize=18)
        plt.xlabel('Lag', fontsize=14)
        plt.ylabel('Autocorrelation', fontsize=14)

        # PACF
        lag_pacf = pacf(train_set.PM2_5, nlags=20, method='ols')

        plt.subplot(212)
        plt.stem(lag_pacf)
        plt.axhline(y=0, linestyle='-', color='black')
        plt.axhline(y=-1.96/np.sqrt(len(train_set)), linestyle='--', color='gray')
        plt.axhline(y=1.96/np.sqrt(len(train_set)), linestyle='--', color='gray')
        plt.title('PACF Plot: PM2.5 Concentration', fontsize=18)
        plt.xlabel('Lag', fontsize=14)
        plt.ylabel('Partial Autocorrelation', fontsize=14)

        plt.tight_layout();
        exog_train = train_set.drop("PM2_5", axis=1)
        exog_test = test_set.drop("PM2_5", axis=1)
        fit_wk = pm.auto_arima(train_set.PM2_5,start_p=1, d=1, start_q=1, max_p=6, max_d=1, max_q=4, 
                                start_P=0, D=1, start_Q=0, max_P=6, max_D=1, max_Q=6, seasonal=True, m=52, trace=True,
                                 error_action='ignore', suppress_warnings=True, stepwise=True)  
        print(fit_wk.summary())
    
        plt.figure(figsize=(12, 8))
        sns.set_style('ticks')

        plt.scatter(train_set.index, train_set.PM2_5, color='steelblue', marker='o')
        plt.plot(train_set.index, fit_wk.predict_in_sample(), color='steelblue', linewidth=3, alpha=0.6)

        fit_test, ci_test = fit_wk.predict(n_periods=test_set.shape[0], return_conf_int=True)
        ci_lower = pd.Series(ci_test[:, 0], index=test_set.index)
        ci_upper = pd.Series(ci_test[:, 1], index=test_set.index)
        plt.scatter(test_set.index, test_set.PM2_5, color='darkred', marker='D')
        plt.plot(test_set.index, fit_wk.predict(n_periods=test_set.shape[0]), color='darkred', linestyle='--', linewidth=3, alpha=0.6)

        plt.title('SARIMA Forecast of Beijing Air Quality', fontsize=18)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('PM2.5 Concentration `(ug / m^3)', fontsize=14)
        plt.axvline(x=location_wk.PM2_5.index[location_wk_index], color='black', linewidth=4, alpha=0.4)
        plt.fill_between(ci_lower.index, ci_lower, ci_upper, color='k', alpha=0.2)
        plt.legend(('Data', 'Forecast', '95% Confidence Interval'), loc='best', prop={'size': 12})
        plt.show();
        print('Training RMSE: %.2f' % np.sqrt(mean_squared_error(train_set.PM2_5, fit_wk.predict_in_sample())))
        print('Testing RMSE: %.2f' % np.sqrt(mean_squared_error(test_set.PM2_5, fit_wk.predict(n_periods=test_set.shape[0]))))

        print("modeling for ",sites[x],"is done\n\n")


# In[20]:
