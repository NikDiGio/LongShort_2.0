#!/usr/bin/env python


# -------------INCIPIT------------------
# The aim of this Data Combinator is to put together the sentiment scores forecasted through the CNN and the Transformers with the financial time series downloaded from the Yahoo Finance API to build the dataset that will be used to assess the investment strategy in the LSTM.
# In particular it has been performed a Left Join of the OHLC prices + Volume table on the sentiment scores since the safest assumption is to have the prices unchanged (due to weekends and holidays). On the other hand a right join would have meant assuming unchanged sentiment scores which is very unlikely in the context of internet.
# ---------------------------------------



import pandas as pd
import random
from scipy import stats
import pandas as pd
import os

parent_directory = os.getcwd()

# In[2]:


# text_data = pd.read_csv('WS_Final_Dataset.csv')
numerical_data = pd.read_csv(parent_directory + '/Input_Data/BABA_financial_data.csv') 
daily_sentiment_finbert = pd.read_csv(parent_directory + '/Output/Transformers/daily_sentiment_Transformers_yiyanghkust_finbert-tone.csv')
daily_sentiment_distilbert = pd.read_csv(parent_directory + '/Output/Transformers/daily_sentiment_Transformers_distilbert-base-uncased.csv')
daily_sentiment_CNN = pd.read_csv(parent_directory + '/Output/CNN/daily_sentiment_CNN_regress.csv')
# no index_col = 0 -> date must be a normal column for merging



# FOR THE REGRESSION TASK 
# ## Join the datasets

daily_sentiment_finbert = daily_sentiment_finbert.rename(columns={'date': 'Date'})
daily_sentiment_distilbert = daily_sentiment_distilbert.rename(columns={'date': 'Date'})
daily_sentiment_CNN = daily_sentiment_CNN.rename(columns={'date': 'Date'})

daily_sentiment = pd.concat([daily_sentiment_finbert['Date'], daily_sentiment_finbert['label'], daily_sentiment_distilbert['label'], daily_sentiment_CNN['label']], axis = 1, keys = ['Date','label_finbert','label_distilbert','label_CNN'])
print(daily_sentiment)



# Left join on the articles' sentiments since I couldn't do assumptions about "Missing" labels
daily_sentiment = daily_sentiment.merge(numerical_data, on='Date', how='left').fillna('Missing')



# Assumption: replaced with the previous value all the missing values that aren't older than the start date of the time series
day1 = numerical_data.loc[0, 'Date']
print(day1)
for row_n in range(1, daily_sentiment.shape[0]):  # Skip first row
    if daily_sentiment['Close'][row_n] == 'Missing' and daily_sentiment['Date'][row_n] >= day1: # Not applying the method on older data points
        daily_sentiment.loc[row_n, 'Open'] = daily_sentiment.loc[row_n - 1, 'Open']
        daily_sentiment.loc[row_n, 'High'] = daily_sentiment.loc[row_n - 1, 'High']
        daily_sentiment.loc[row_n, 'Low'] = daily_sentiment.loc[row_n - 1, 'Low']
        daily_sentiment.loc[row_n, 'Close'] = daily_sentiment.loc[row_n - 1, 'Close']
        daily_sentiment.loc[row_n, 'Adj Close'] = daily_sentiment.loc[row_n - 1, 'Adj Close']
        daily_sentiment.loc[row_n, 'Volume'] = daily_sentiment.loc[row_n - 1, 'Volume']





# Excluding the missing rows
combined_dataset = daily_sentiment.loc[daily_sentiment['Close'] != 'Missing',:]

# Exporting to csv
colnames = combined_dataset.columns.tolist()
if 'label_CNN' in colnames:
    combined_dataset.to_csv(parent_directory + '/Input_Data/combined_dataset_4_regression_with_CNN.csv')
else:
	combined_dataset.to_csv(parent_directory + '/Input_Data/combined_dataset_4_regression.csv')


