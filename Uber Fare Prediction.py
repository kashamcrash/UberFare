#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Uber Fare Prediction
# Design an algorithm that will predict the fare to be charged for a passenger

# Credentials: kasham1991@gmail.com / Karan Sharma
# # Google Drive Link (1.6 GB) : https://drive.google.com/open?id=1Yp_1LWg4rtBj6ezbu6AqGRO8LpFUdYiD


# In[2]:


# Importing the required libraries
import numpy as np 
import pandas as pd 


# In[3]:


# Loading the dataset
# The training sheet is massive, hence calling out only 9 lakh rows
train = pd.read_csv("C:\\Datasets\\Uber_train.csv", nrows = 900000)
test = pd.read_csv("C:\\Datasets\\Uber_test.csv")


# In[4]:


# Looking at the dataset
print (train.shape)
print (train.columns)
print (test.shape)
print (test.columns)


# In[5]:


# Looking at the dtypes
# We need to convert pickup_datetime into date time in python
train.info()
# train.head()


# In[6]:


# Converting to date time format
train["pickup_datetime"] = pd.to_datetime(train['pickup_datetime'])
train.head()
# train.info()


# In[7]:


# Basic Statistics 
# How is minimum fare_amount negative?
# How is latitude and longitude in thousands?
# Latitude ranges from +90 to -90 while longitude ranges from -180 to +180
# We need to treat these values accordingly
train.describe()
# train.describe().T


# In[8]:


# Checking for null values
# There are null values in longitude/latitude rows
train.isnull().sum()


# In[9]:


# Dropping null values
train.dropna(inplace = True)
train.isnull().sum()


# In[10]:


# Lets visualize the negative and extreme values
# Seaborn distplot shows a histogram distrubution with a line on it
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


sns.distplot(train['fare_amount'])
# Negative fair amount


# In[12]:


sns.distplot(train['pickup_latitude'])
# Greater than 1000 on either side


# In[13]:


sns.distplot(train['pickup_longitude'])
# Greater than 1000 on either side


# In[14]:


sns.distplot(train['dropoff_longitude'])
# Goes all the way up to negative 3000


# In[15]:


sns.distplot(train['dropoff_latitude'])
# This indicates to much noise in the data


# In[16]:


# Looking at the range of latitude and longitude
print(test["dropoff_latitude"].min()) # drop_off latitude min
print(test["dropoff_latitude"].max()) # drop_off latitude max
print(test["dropoff_longitude"].min()) # drop_off longitude min
print(test["dropoff_longitude"].max()) # drop_off longitude max
print(test["pickup_latitude"].min()) # pickup latitude min
print(test["pickup_latitude"].max()) # pickup latitude max
print(test["pickup_longitude"].min()) # pickup longitude min
print(test["pickup_longitude"].max()) # pickup longitude max


# In[17]:


# Removing all negative values in latitude and longitude
# Setting the range for noisy data 
min_longitude = -74.263242
min_latitude = 40.573143
max_longitude = -72.986532
max_latitude = 41.709555

Noise = train[(train["dropoff_latitude"]<min_latitude) | (train["pickup_latitude"]<min_latitude) | (train["dropoff_longitude"]<min_longitude) | (train["pickup_longitude"]<min_longitude) | (train["dropoff_latitude"]>max_latitude) | (train["pickup_latitude"]>max_latitude) | (train["dropoff_longitude"]>max_longitude) | (train["pickup_longitude"]>max_longitude) ]
# Noise.shape
train.drop(Noise.index, inplace = True)
train.shape


# In[18]:


# Removing negative fare value
train = train[train['fare_amount']>0]
train.shape


# In[19]:


# The fare price of Uber is subject to surge during high demand
# Sure occurs druing peak and rush hours
# Lets extract the different days/weeks/time/months that are subject to surge in seperate columns
# Lambda is a one time use function

import calendar
train['day'] = train['pickup_datetime'].apply(lambda x: x.day)
train['hour'] = train['pickup_datetime'].apply(lambda x: x.hour)
train['weekday'] = train['pickup_datetime'].apply(lambda x: calendar.day_name[x.weekday()])
train['month'] = train['pickup_datetime'].apply(lambda x: x.month)
train['year'] = train['pickup_datetime'].apply(lambda x: x.year)
train.head()


# In[20]:


# Coverting days of the week to numbers with the map function
train.weekday = train.weekday.map({'Sunday': 0,'Monday': 1,'Tuesday': 2,'Wednesday': 3,'Thursday': 4,'Friday': 5,'Saturday': 6})
# train.weekday
train.info()


# In[21]:


# Removing redundant columns such as key and pickup_datetime
train.drop(["key","pickup_datetime"], axis = 1, inplace=True)
train.info()


# In[22]:


# Due to a very high number of floats and integers
# Lets standardize the dataset and remove NaN
# Standardization involves shifting the distribution of each data point to a mean of 0 and an SD of 1
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
h = pd.DataFrame(sc_x.fit_transform(train))
l = train.dropna(axis = 1)


# In[23]:


# Splitting the Dataset
# Since we have to predict the amount of fare, this becomes our output variable 
x = l.drop("fare_amount", axis = 1)
y = l['fare_amount']


# In[24]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# x_train.head()
# x_test.head()
# x_train.shape
# x_test.shape


# In[26]:


# Since the predicted variable is continous, linear regression will be a suitable model
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(x_train, y_train)


# In[27]:


# Looking at the predicted array
y_predict = model1.predict(x_test)
y_predict


# In[28]:


# Calculating the rmse for linear Regression model
# The closer the value to 1, more the accuracy
from sklearn.metrics import mean_squared_error
model1_rmse = np.sqrt(mean_squared_error(y_predict, y_test))
print("RMSE value for Linear regression is", model1_rmse)


# In[30]:


# Trying Randomn Forest Classifier
from sklearn.ensemble import RandomForestRegressor
rfmodel = RandomForestRegressor(n_estimators = 10, random_state = 42)
rfmodel.fit(x_train,y_train)
rfmodel_pred= rfmodel.predict(x_test)


# In[31]:


rfmodel_rmse = np.sqrt(mean_squared_error(rfmodel_pred, y_test))
print("RMSE value for Random forest regression is ", rfmodel_rmse)


# In[32]:


# RSME for Random Forest is closer to 1 
# RFC will be the preferred model
# Thank You

