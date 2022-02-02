#!/usr/bin/env python
# coding: utf-8

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
import warnings
warnings.filterwarnings("ignore")

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

# In[3]:


print(sites.shape)

location = np.empty(12, dtype=object)


# In[9]:


import numpy as np
myseed = 7
np.random.seed(myseed)


# In[12]:


import os.path

city_names = []
for x in range(12):
    location[x] = pd.read_csv(sites[x])
    city_names.append(sites[x].split("_")[2])
    location[x].shape


# In[13]:


city_names


# In[14]:


# convert to datetime index
for x in range(12):
    location[x].index = pd.to_datetime(location[x][['year', 'month', 'day', 'hour']])
    print(location[x].index[0])
    print(location[x].index[-1])


# In[15]:


for x in range(12):
    location[x].columns
    location[x] = location[x].rename({'PM2.5': 'PM2_5'}, axis=1)  # new method
    print(location[x].columns)
    # descriptive stats
    print(location[x].PM2_5.describe())
    print(location[x].isna().sum())
    print("\n\n After handling null values")
    location[x].drop(["No", "year", "month", "day", "hour", "PM10"], axis=1, inplace=True)
    location[x]['PM2_5'] = location[x]['PM2_5'].interpolate(method="linear")
    location[x]['SO2'] = location[x]['SO2'].interpolate(method="linear")
    location[x]['NO2'] = location[x]['NO2'].interpolate(method="linear")
    location[x]['CO'] = location[x]['CO'].interpolate(method="linear")
    location[x]['O3'] = location[x]['O3'].interpolate(method="linear")
    location[x]['TEMP'] = location[x]['TEMP'].interpolate(method="linear")
    location[x]['PRES'] = location[x]['PRES'].interpolate(method="linear")
    location[x]['DEWP'] = location[x]['DEWP'].interpolate(method="linear")
    location[x]['RAIN'] = location[x]['RAIN'].interpolate(method="linear")
    location[x]['WSPM'] = location[x]['WSPM'].interpolate(method="linear")
    # descriptive stats after handling null values
    print(location[x].PM2_5.describe())
    print(location[x].isna().sum())
    print("\n\n")


# In[20]:


for x in range(12):
    # convert frequency to weeks
    
    location_wk = location[x].resample('W').mean()
    print('week frequency shape:', location_wk.shape)
    print('PM2.5 missing values:', location_wk.isna().sum()['PM2_5'])
    plt.figure(figsize=(10, 5))
    plt.plot(location[x].PM2_5)
    plt.title(('Beijing Air Quality in {0} site: PM2.5 Concentration from 2013 - 2017').format(city_names[x]), fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('PM2.5 Concentration (ug / m^3)', fontsize=14);
    plt.style.use('seaborn-dark-palette')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.show()
    
    decomposition = sm.tsa.seasonal_decompose(location_wk.PM2_5, model='additive', extrapolate_trend='freq')
    fig = decomposition.plot()
    plt.show()
    
    location_aft = adfuller(location_wk.PM2_5, autolag='AIC')
    output = pd.Series(location_aft[0:4], index=['test statistic', 'pvalue', 'number of lags used', 'number of observations'])
    print(output)

    location_diff = location_wk.diff().dropna()
    print(location_diff)

    # train-test split
    pct_train = 0.80
    location_wk_index = round(len(location_wk) * pct_train)
    train_set, test_set = location_wk[:location_wk_index], location_wk[location_wk_index:]

    # ACF
    lag_acf = acf(location_diff.PM2_5, nlags=20)

    plt.figure(figsize=(12, 10))
    plt.subplot(211)
    plt.stem(lag_acf)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=-1.96/np.sqrt(len(location_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(location_diff)), linestyle='--', color='gray')
    plt.title('ACF Plot: PM2.5 Concentration', fontsize=18)
    plt.xlabel('Lag', fontsize=14)
    plt.ylabel('Autocorrelation', fontsize=14)
    
    #plt.show()

    # PACF
    lag_pacf = pacf(location_diff.PM2_5, nlags=20, method='ols')

    plt.subplot(212)
    plt.stem(lag_pacf)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=-1.96/np.sqrt(len(location_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(location_diff)), linestyle='--', color='gray')
    plt.title('PACF Plot: PM2.5 Concentration', fontsize=18)
    plt.xlabel('Lag', fontsize=14)
    plt.ylabel('Partial Autocorrelation', fontsize=14)

    plt.tight_layout();
    
    plt.show()

    #exogenous inputs
    exog_train = train_set.drop("PM2_5", axis=1)
    exog_test = test_set.drop("PM2_5", axis=1)

    #Order of AR(p) and MA(q)
    ar_p = 3          # this is the maximum degree specification chosen from pacf graph
    ma_q = 1          # this is the lag polynomial specification chosen from acf graph

    print("\n\nBUILDING MODEL...(Executing wait for the model to finish the build)")
    # Fit the model
    mod = sm.tsa.statespace.SARIMAX(train_set['PM2_5'], exog_train, order=(ar_p,1,ma_q), seasonal_order=(1,1,1,52))
    fit_res = mod.fit(disp=False)
    print(fit_res.summary())

    train_pred = fit_res.predict(start = 0,end = (train_set.shape[0]-1),exog = exog_train)[0:]
    print('SARIMAX model training MSE:{}'.format(mean_squared_error(train_set["PM2_5"],train_pred)))
    print('SARIMAX model training RMSE:{}'.format(np.sqrt(mean_squared_error(train_set["PM2_5"],train_pred))))

    pd.DataFrame({'train':train_set["PM2_5"],'pred':train_pred}).plot(title="SARIMAX Forecast Train set", 
                                                  xlabel="Date", 
                                                  ylabel="Weekly PM2.5 Concentration `(ug / m^3)", 
                                                  fontsize=14
                                                 )

    pred = fit_res.predict(start = train_set.shape[0],end = (train_set.shape[0]+test_set.shape[0]-1),exog = exog_test)[0:]
    print('SARIMAX model test MSE:{}'.format(mean_squared_error(test_set["PM2_5"],pred)))
    print('SARIMAX model test RMSE:{}'.format(np.sqrt(mean_squared_error(test_set["PM2_5"],pred))))

    pd.DataFrame({'test':test_set["PM2_5"],'pred':pred}).plot(title="SARIMAX Forecast Test set", 
                                                  xlabel="Date", 
                                                  ylabel="Weekly PM2.5 Concentration `(ug / m^3)", 
                                                  fontsize=14
                                                 )
    plt.show()
    print("====================================================================================================")


# In[ ]:




