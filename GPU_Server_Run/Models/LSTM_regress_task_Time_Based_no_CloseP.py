#!/usr/bin/env python3

# -------------INCIPIT------------------
# This script's job is to forecast stock prices and calculate the relative financial performance compared to the benchmark strategy of Buy and Hold. 
# Specifically this code will consider a time-based dataset splitting with no past Close prices in the feature set.
# ---------------------------------------

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
import bs4
import os

time_lag_max = 5 # Look-back period to be remembered from the LSTM

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
print(parent_directory)


def calculate_cumproduct(arr, capital=1000):
    df = pd.DataFrame(arr)  # Convert the prices to a DataFrame
    returns = (df - df.shift(1)) / df.shift(1)
    returns = returns.iloc[1:]  # Skip the first row since it contains NaN after the shift
    # Calculate cumulative product of returns
    cumulative_product = (1 + returns).cumprod()

    capital_evolution = capital * cumulative_product
    total_return = ((cumulative_product.iloc[-1] - 1) * 100).values[0] # Extract the float value
    
    return total_return, capital_evolution.values


def calculate_cumproduct_strategy(arr_true, arr_strat, capital=1000):
    df_true = pd.DataFrame(arr_true)  # Convert the prices to a DataFrame
    df_strat = pd.DataFrame(arr_strat)  # Convert the prices to a DataFrame
    returns = []
    
    for i in range(df_true.shape[0]):
        if i > 0:  # Skip first row
            if df_strat.iloc[i, 0] >= df_true.iloc[i-1, 0]:
                returns.append((df_true.iloc[i, 0] - df_true.iloc[i-1, 0]) / df_true.iloc[i-1, 0]) # Go Long
            else:
                returns.append(-((df_true.iloc[i, 0] - df_true.iloc[i-1, 0]) / df_true.iloc[i-1, 0])) # Go Short

    returns = pd.Series(returns)
    cumulative_product = (1 + returns).cumprod()  # Calculate cumulative product of returns
    
    capital_evolution = capital * cumulative_product
    total_return = (cumulative_product.iloc[-1] - 1) * 100
    
    return total_return, capital_evolution


def LSTM_df_builder(df, time_lag_max):
    column_names = df.columns.tolist()

    for name in column_names:
        for i in range(2, time_lag_max + 1): # from 2 because I already shifted by 1
            new_colname = f'{name} t-{i}'
            df[new_colname] = np.nan
            plc_hld = []
            for row in range(i - 1, df.shape[0]):
                if row - i + 1 >= 0:  # Check if the index is non-negative
                    df.loc[row, new_colname] = df.loc[(row - i + 1), name]

    return df



# Hyperparameter for deciding the foreacst time frequency (in this case daily) 
num_time_steps = 1


# The product of all the dimensions of a tensor must remain the same but is not always the case when you impose num_time_steps.
# Therefore we truncate the data to respect the rule. 
def truncate_for_num_time_steps(arr, num_time_steps):
    rest = arr.shape[0] % num_time_steps
    if rest != 0:
        arr = arr[:-rest]  # truncate at the end
    return arr
# As long as the truncation is contained in the number we accept it over padding to avoid inserting any bias in the model training

# Load your dataset and preprocess it
data_pre = pd.read_csv(parent_directory + '/Input_Data/combined_dataset_4_regression_with_CNN.csv', index_col = 0)

print('data \n', data_pre)

# CHOSEN SPLITTING STRATEGY: "time-based split" approach: In time series analysis, preserving the temporal order of data is crucial because future values depend on past values
train_indices = range(0, int(data_pre.shape[0] * 0.80) - time_lag_max) # - time_lag_max to consider the shifting and the Feature Engineering made below
test_indices = range(int(data_pre.shape[0] * 0.80) - time_lag_max, data_pre.shape[0] - time_lag_max)

print("Train indices ", train_indices)
print("Test indices ", test_indices)

# Build the custom indices for the prices and returns
# Prices
custom_indices = data_pre.loc[test_indices, 'Date'].values
custom_indices = pd.to_datetime(custom_indices)  # Convert indices to timestamps

# Returns
custom_indices_ret = data_pre.loc[test_indices[1:], 'Date'].values
custom_indices_ret = pd.to_datetime(custom_indices_ret)  # Convert indices to timestamps


# Build the datasets
# Build the datasets using list comprehension
data = data_pre[[column for column in data_pre.columns if column not in ['label_CNN', 'label_finbert', 'label_distilbert', 'Date']]]
data_distilbert = data_pre[[column for column in data_pre.columns if column not in ['label_CNN', 'label_finbert', 'Date']]]
data_finbert = data_pre[[column for column in data_pre.columns if column not in ['label_CNN', 'label_distilbert', 'Date']]]
data_CNN = data_pre[[column for column in data_pre.columns if column not in ['label_finbert', 'label_distilbert', 'Date']]]


# 'Close' is the column for closing prices
target = data['Close']
features_no_SA = data.drop(columns=['Close'])
features_distilbert = data_distilbert.drop(columns=['Close'])
features_finbert = data_finbert.drop(columns=['Close'])
features_CNN = data_CNN.drop(columns=['Close'])


print(target)
print(features_no_SA)
print(features_distilbert)
print(features_finbert)
print(features_CNN)

# Shift the features one day ahead
features_no_SA = features_no_SA.shift(1)
features_distilbert = features_distilbert.shift(1)
features_finbert = features_finbert.shift(1)
features_CNN = features_CNN.shift(1)


# Remove the first row of NaN from the target variable
target = target.iloc[1:]

# Drop the first row from the shifted features to align the data
features_no_SA = features_no_SA.iloc[1:]
features_distilbert = features_distilbert.iloc[1:]
features_finbert = features_finbert.iloc[1:]
features_CNN = features_CNN.iloc[1:]


# Indeces did not start from 0 so I reset it
target = target.reset_index(drop = True)
features_no_SA = features_no_SA.reset_index(drop = True)
features_distilbert = features_distilbert.reset_index(drop = True)
features_finbert = features_finbert.reset_index(drop = True)
features_CNN = features_CNN.reset_index(drop = True)


print(target)
print(features_no_SA)
print(features_distilbert)
print(features_finbert)
print(features_CNN)


# Applying the custom function
features_no_SA = LSTM_df_builder(features_no_SA, time_lag_max)
features_distilbert = LSTM_df_builder(features_distilbert, time_lag_max)
features_finbert = LSTM_df_builder(features_finbert, time_lag_max)
features_CNN = LSTM_df_builder(features_CNN, time_lag_max)


# Drop the NaN
target = target.dropna()
features_no_SA = features_no_SA.dropna()
features_distilbert = features_distilbert.dropna()
features_finbert = features_finbert.dropna()
features_CNN = features_CNN.dropna()

print(features_no_SA)
print(features_distilbert)
print(features_finbert)
print(features_CNN)

# Reset again the indeces
target = target.reset_index(drop = True)
features_no_SA = features_no_SA.reset_index(drop = True)
features_distilbert = features_distilbert.reset_index(drop = True)
features_finbert = features_finbert.reset_index(drop = True)
features_CNN = features_CNN.reset_index(drop = True)

print('Second index resetting')
print(target)
print(features_no_SA)
print(features_distilbert)
print(features_finbert)
print(features_CNN)


# Scale the features using Min-Max scaling
scaler = MinMaxScaler()
scaled_features_no_SA = scaler.fit_transform(features_no_SA)
scaled_features_distilbert = scaler.fit_transform(features_distilbert)
scaled_features_finbert = scaler.fit_transform(features_finbert)
scaled_features_CNN = scaler.fit_transform(features_CNN)




# Create an LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape,  activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


X_train_no_SA, X_test_no_SA = scaled_features_no_SA[train_indices], scaled_features_no_SA[test_indices]
y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]


print("X_train_no_SA \n", X_train_no_SA)
print("y_train \n", y_train)

print('Shape X_train_no_SA:',X_train_no_SA.shape)
print('Shape X_test_no_SA:',X_test_no_SA.shape)
print('Shape y_train:',y_train.shape)
print('Shape y_test:',y_test.shape)

X_train_no_SA, X_test_no_SA = truncate_for_num_time_steps(X_train_no_SA,num_time_steps), truncate_for_num_time_steps(X_test_no_SA,num_time_steps)
y_train, y_test = truncate_for_num_time_steps(y_train,num_time_steps), truncate_for_num_time_steps(y_test,num_time_steps)

print('Shape X_train_no_SA:',X_train_no_SA.shape)
print('Shape X_test_no_SA:',X_test_no_SA.shape)
print('Shape y_train:',y_train.shape)
print('Shape y_test:',y_test.shape)


y_train = y_train.values
y_test = y_test.values

# Assuming you have reshaped your data as described earlier
X_train_reshaped_no_SA = X_train_no_SA.reshape(-1, num_time_steps, X_train_no_SA.shape[1])
X_test_reshaped_no_SA = X_test_no_SA.reshape(-1, num_time_steps, X_test_no_SA.shape[1])
print('Shape X_train_reshaped:',X_train_reshaped_no_SA.shape)
print('Shape X_test_reshaped:',X_test_reshaped_no_SA.shape)

y_train_reshaped = y_train.reshape(-1, num_time_steps, 1) # Single Target Prediction Task -> Close
print('Shape y_train_reshaped:',y_train_reshaped.shape)

# Convert the data to tensors
X_train_tensor_no_SA = tf.constant(X_train_reshaped_no_SA, dtype=tf.float32)
X_test_tensor_no_SA = tf.constant(X_test_reshaped_no_SA, dtype=tf.float32)
print('Shape X_train_tensor:',X_train_tensor_no_SA.shape)
print('Shape X_test_tensor:',X_test_tensor_no_SA.shape)

y_train_tensor = tf.constant(y_train_reshaped, dtype=tf.float32)
print('Shape y_train_tensor:',y_train_tensor.shape)


# Create and train the model
model_no_SA = create_lstm_model((num_time_steps, X_train_no_SA.shape[1])) # In Keras, when defining the input shape for the first layer of the model, you need to specify (at least) the number of features (i.e., the    number of columns in your input data).

history_no_SA = model_no_SA.fit(X_train_tensor_no_SA, y_train_tensor, epochs=50, batch_size=32, verbose=0)

# Display a summary of the model
model_no_SA.summary()
	
# Evaluate the model
y_pred_no_SA = model_no_SA.predict(X_test_tensor_no_SA)

# Flattening the y pred to 1d to match y test dimensions
y_pred_no_SA = y_pred_no_SA.reshape(-1,)

print(y_pred_no_SA)

    
all_mse_values_no_SA = history_no_SA.history['loss']

true_prices = y_test
predicted_prices_no_SA = y_pred_no_SA

# Capital evolution over time and Total Returns

total_return_true, capital_evolution_true = calculate_cumproduct(y_test)
total_return_no_SA, capital_evolution_no_SA = calculate_cumproduct_strategy(y_test, y_pred_no_SA)

ret_true = capital_evolution_true
print('capital_evolution_true shape', capital_evolution_true.shape)
ret_no_SA = capital_evolution_no_SA
print('capital_evolution_no_SA shape', capital_evolution_no_SA.shape)

total_returns_true = total_return_true
total_returns_no_SA = total_return_no_SA

# -------------------------------------------------------------------------------------------------------------------

X_train_finbert, X_test_finbert = scaled_features_finbert[train_indices], scaled_features_finbert[test_indices]

print('Shape X_train_finbert:',X_train_finbert.shape)
print('Shape X_test_finbert:',X_test_finbert.shape)

X_train_finbert, X_test_finbert = truncate_for_num_time_steps(X_train_finbert,num_time_steps), truncate_for_num_time_steps(X_test_finbert,num_time_steps)


print('Shape X_train_finbert:',X_train_finbert.shape)
print('Shape X_test_finbert:',X_test_finbert.shape)



# Assuming you have reshaped your data as described earlier
X_train_reshaped_finbert = X_train_finbert.reshape(-1, num_time_steps, X_train_finbert.shape[1])
X_test_reshaped_finbert = X_test_finbert.reshape(-1, num_time_steps, X_test_finbert.shape[1])
print('Shape X_train_reshaped:',X_train_reshaped_finbert.shape)
print('Shape X_test_reshaped:',X_test_reshaped_finbert.shape)


# Convert the data to tensors
X_train_tensor_finbert = tf.constant(X_train_reshaped_finbert, dtype=tf.float32)
X_test_tensor_finbert = tf.constant(X_test_reshaped_finbert, dtype=tf.float32)
print('Shape X_train_tensor:',X_train_tensor_finbert.shape)
print('Shape X_test_tensor:',X_test_tensor_finbert.shape)



# Create and train the model
model_finbert = create_lstm_model((num_time_steps, X_train_finbert.shape[1])) # In Keras, when defining the input shape for the first layer of the model, you need to specify (at least) the number of features (i.e., the    number of columns in your input data).

history_finbert = model_finbert.fit(X_train_tensor_finbert, y_train_tensor, epochs=50, batch_size=32, verbose=0)

# Display a summary of the model
model_finbert.summary()
	
# Evaluate the model
y_pred_finbert = model_finbert.predict(X_test_tensor_finbert)    

# Flattening the y pred to 1d to match y test dimensions
y_pred_finbert = y_pred_finbert.reshape(-1,)

print(y_pred_finbert)

all_mse_values_finbert = history_finbert.history['loss']

# mse_values.append(mse)
predicted_prices_finbert = y_pred_finbert


# Capital evolution over time and Total Returns

total_return_finbert, capital_evolution_finbert = calculate_cumproduct_strategy(y_test, y_pred_finbert)

ret_finbert = capital_evolution_finbert
print('capital_evolution_finbert shape', capital_evolution_finbert.shape)

total_returns_finbert = total_return_finbert


# -------------------------------------------------------------------------------------------------------------------


X_train_distilbert, X_test_distilbert = scaled_features_distilbert[train_indices], scaled_features_distilbert[test_indices]

print('Shape X_train_distilbert:',X_train_distilbert.shape)
print('Shape X_test_distilbert:',X_test_distilbert.shape)


X_train_distilbert, X_test_distilbert = truncate_for_num_time_steps(X_train_distilbert,num_time_steps), truncate_for_num_time_steps(X_test_distilbert,num_time_steps)


print('Shape X_train_distilbert:',X_train_distilbert.shape)
print('Shape X_test_distilbert:',X_test_distilbert.shape)

   

# Assuming you have reshaped your data as described earlier
X_train_reshaped_distilbert = X_train_distilbert.reshape(-1, num_time_steps, X_train_distilbert.shape[1])
X_test_reshaped_distilbert = X_test_distilbert.reshape(-1, num_time_steps, X_test_distilbert.shape[1])
print('Shape X_train_reshaped:',X_train_reshaped_distilbert.shape)
print('Shape X_test_reshaped:',X_test_reshaped_distilbert.shape)


# Convert the data to tensors
X_train_tensor_distilbert = tf.constant(X_train_reshaped_distilbert, dtype=tf.float32)
X_test_tensor_distilbert = tf.constant(X_test_reshaped_distilbert, dtype=tf.float32)
print('Shape X_train_tensor:',X_train_tensor_distilbert.shape)
print('Shape X_test_tensor:',X_test_tensor_distilbert.shape)



# Create and train the model
model_distilbert = create_lstm_model((num_time_steps, X_train_distilbert.shape[1])) # In Keras, when defining the input shape for the first layer of the model, you need to specify (at least) the number of features (i.e., the    number of columns in your input data).

history_distilbert = model_distilbert.fit(X_train_tensor_distilbert, y_train_tensor, epochs=50, batch_size=32, verbose=0)

# Display a summary of the model
model_distilbert.summary()
	
# Evaluate the model
y_pred_distilbert = model_distilbert.predict(X_test_tensor_distilbert)

# Flattening the y pred to 1d to match y test dimensions
y_pred_distilbert = y_pred_distilbert.reshape(-1,)

print(y_pred_distilbert)

all_mse_values_distilbert = history_distilbert.history['loss']

# mse_values.append(mse)
predicted_prices_distilbert = y_pred_distilbert
    
# Capital evolution over time and Total Returns

total_return_distilbert, capital_evolution_distilbert = calculate_cumproduct_strategy(y_test, y_pred_distilbert)

ret_distilbert = capital_evolution_distilbert
print('capital_evolution_distilbert shape', capital_evolution_distilbert.shape)

total_returns_distilbert = total_return_distilbert

# -------------------------------------------------------------------------------------------------------------------



X_train_CNN, X_test_CNN = scaled_features_CNN[train_indices], scaled_features_CNN[test_indices]

print('Shape X_train_CNN:',X_train_CNN.shape)
print('Shape X_test_CNN:',X_test_CNN.shape)


X_train_CNN, X_test_CNN = truncate_for_num_time_steps(X_train_CNN,num_time_steps), truncate_for_num_time_steps(X_test_CNN,num_time_steps)


print('Shape X_train_CNN:',X_train_CNN.shape)
print('Shape X_test_CNN:',X_test_CNN.shape)

   

# Assuming you have reshaped your data as described earlier
X_train_reshaped_CNN = X_train_CNN.reshape(-1, num_time_steps, X_train_CNN.shape[1])
X_test_reshaped_CNN = X_test_CNN.reshape(-1, num_time_steps, X_test_CNN.shape[1])
print('Shape X_train_reshaped:',X_train_reshaped_CNN.shape)
print('Shape X_test_reshaped:',X_test_reshaped_CNN.shape)


# Convert the data to tensors
X_train_tensor_CNN = tf.constant(X_train_reshaped_CNN, dtype=tf.float32)
X_test_tensor_CNN = tf.constant(X_test_reshaped_CNN, dtype=tf.float32)
print('Shape X_train_tensor:',X_train_tensor_CNN.shape)
print('Shape X_test_tensor:',X_test_tensor_CNN.shape)



# Create and train the model
model_CNN = create_lstm_model((num_time_steps, X_train_CNN.shape[1])) # In Keras, when defining the input shape for the first layer of the model, you need to specify (at least) the number of features (i.e., the    number of columns in your input data).

history_CNN = model_CNN.fit(X_train_tensor_CNN, y_train_tensor, epochs=50, batch_size=32, verbose=0)

# Display a summary of the model
model_CNN.summary()
	
# Evaluate the model
y_pred_CNN = model_CNN.predict(X_test_tensor_CNN)

# Flattening the y pred to 1d to match y test dimensions
y_pred_CNN = y_pred_CNN.reshape(-1,)

print(y_pred_CNN)

all_mse_values_CNN = history_CNN.history['loss']

# mse_values.append(mse)
predicted_prices_CNN = y_pred_CNN


# Capital evolution over time and Total Returns

total_return_CNN, capital_evolution_CNN = calculate_cumproduct_strategy(y_test, y_pred_CNN)

ret_CNN = capital_evolution_CNN
print('capital_evolution_CNN shape', capital_evolution_CNN.shape)

total_returns_CNN = total_return_CNN

# -------------------------------------------------------------------------------------------------------------------




# Plot True and Predicted Prices for the average of all folds
plt.figure(figsize=(12, 6))
plt.plot(custom_indices, true_prices, label='True Prices', alpha=0.5)
plt.plot(custom_indices, predicted_prices_no_SA, label='Predicted Prices no SA', alpha=0.5)
plt.plot(custom_indices, predicted_prices_finbert, label='Predicted Prices finbert', alpha=0.5)
plt.plot(custom_indices, predicted_prices_distilbert, label='Predicted Prices distilbert', alpha=0.5)
plt.plot(custom_indices, predicted_prices_CNN, label='Predicted Prices CNN', alpha=0.5)
plt.legend()
plt.title(f'True vs. Predicted Prices time-based split')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.savefig(parent_directory + '/Output/LSTM/with_memory/time_based_splitting/without_past_Close_prices/True vs. Predicted Prices time-based split_no_Close.png')
plt.show()

# Plot Mean Squared Error for the average of all folds over the number of epochs
plt.figure(figsize=(12, 6))
plt.plot(all_mse_values_no_SA, label='mse_values_no_SA', alpha=0.5)
plt.plot(all_mse_values_finbert, label='mse_values_finbert', alpha=0.5)
plt.plot(all_mse_values_distilbert, label='mse_values_distilbert', alpha=0.5)
plt.plot(all_mse_values_CNN, label='mse_values_CNN', alpha=0.5)
plt.legend()
plt.title(f'MSE Over Epochs time-based split')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.grid(True)
plt.savefig(parent_directory + '/Output/LSTM/with_memory/time_based_splitting/without_past_Close_prices/MSE Over Epochs time-based split_no_Close.png')
plt.show()


print('ret_true shape',ret_true.shape)

# Plot the evolution of 1000 euros invested with the following strategies (for the average of all folds over the number of epochs)
plt.figure(figsize=(12, 6))
plt.plot(custom_indices_ret, ret_true, label='Long only', alpha=0.5)
plt.plot(custom_indices_ret, ret_no_SA, label='LSTM no SA', alpha=0.5)
plt.plot(custom_indices_ret, ret_finbert, label='LSTM finbert', alpha=0.5)
plt.plot(custom_indices_ret, ret_distilbert, label='LSTM distilbert', alpha=0.5)
plt.plot(custom_indices_ret, ret_CNN, label='LSTM CNN', alpha=0.5)
plt.legend()
plt.title(f'Evolution of 1000$ invested with the following strategies time-based split')
plt.xlabel('Time')
plt.ylabel('Money')
plt.grid(True)
plt.savefig(parent_directory + '/Output/LSTM/with_memory/time_based_splitting/without_past_Close_prices/evolution_1000$_time-based_split_no_Close.png')
plt.show()



confronto = pd.DataFrame(index=['Long only', 'LSTM no SA', 'LSTM finbert', 'LSTM distilbert', 'LSTM CNN'])


# Building the dataframe of total returns
colname = f'Total returns'
confronto[colname] = [total_returns_true, 
                     total_returns_no_SA, 
                     total_returns_finbert, 
                     total_returns_distilbert, 
                     total_returns_CNN]


confronto.to_csv(parent_directory + '/Output/LSTM/with_memory/time_based_splitting/without_past_Close_prices/confronto time-based split_no_Close.csv')

# Modify the color_format function to handle Series objects
def color_format(val):
    if isinstance(val, pd.Series):
        # If val is a Series, apply the color_format function element-wise
        return val.apply(color_format)

    if val >= 0:
        # Green shades for positive values
        color = f'hsl(120, 50%, {max(70 - (val * 10), 0)}%)'  # Fixed hue (green), varying lightness capped at 0%
        font_color = 'darkgreen'
    else:
        # Red shades for negative values
        color = f'hsl(0, 50%, {max(70 - (val * 10), 0)}%)'  # Fixed hue (red), varying lightness capped at 0%
        font_color = 'darkred'

    background_color = f'background-color: {color};'
    font_color = f'color: {font_color};'

    # Format the values as percentage and rounded figures
    formatted_val = f'{val:.2f}%'

    return f'{background_color} {font_color}', formatted_val

# Apply the formatting function to the DataFrame
styled_confronto = confronto.style.applymap(lambda x: color_format(x)[0])
styled_confronto = styled_confronto.format(lambda x: color_format(x)[1])

# Print the html code behind the dataframe for debugging
print(styled_confronto.render())

# Save the styled DataFrame as an HTML file to keep formatting
styled_confronto.to_html(parent_directory + '/Output/LSTM/with_memory/time_based_splitting/without_past_Close_prices/styled_confronto_total_returns_time-based_plit_no_Close.html')
#
