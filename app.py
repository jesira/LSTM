import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import investpy
from datetime import date
from keras.models import load_model
import streamlit as st

#collecting data from investing.com


st.title = ('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'SGBD')

# user_input = 'SGBD'
country = 'Bangladesh'
start_date = '01/01/2010'
end_date = '01/01/2020'
today = date.today()
today = today.strftime('%d/%m/%Y')
df = investpy.get_stock_historical_data(stock=user_input,
                                        country=country,
                                        from_date=start_date,
                                        to_date=today)


#describing data
# st.subheader('Data from 2010 - 2019')
# st.write(df.describe())

#Visualization 
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize= (12,8))
plt.plot(df['Close'])
st.pyplot(fig)

df = df['Close']
l = list(df.index)
print(" DATTTTTTTTTEEEEEEEEEE : ",len(l))
### LSTM are sensitive to the scale of the data. so applying minmax scaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(np.array(df).reshape(-1,1)) # -1 = row not specified column 1

##splitting dataset size
training_size=int(len(df)*0.70)
test_size=len(df)-training_size
train_data,test_data=df[0:training_size],df[training_size:len(df)]

#creating dataset
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
# # print(X_train.shape)
# # print(y_train.shape)
# # print(X_test.shape)
# # print(y_test.shape)

# # reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

###loading model 
model = load_model('my_lstm_model')


#Testing part 
test_predict=model.predict(X_test)
scaler = scaler.scale_
scale_factor = 1/scaler
test_predict = test_predict * scale_factor
y_test = y_test * scale_factor

#Final Graph 
st.subheader('Predictions vs Original')
# Creating first axes for the figure

fig2 = plt.figure(figsize=(12,8))
ax1 = fig2.add_axes([1, 1, 1, 1])
test_data_dates = l[-y_test.shape[0]:]
ax1.plot(test_data_dates, y_test, label='Actual data')
ax1.plot(test_data_dates, test_predict,'r', label='Predicted data')

# plt.plot(y_test, 'b', label = 'Original Price')
# plt.plot(test_predict, 'r', label = 'Predicted Price')
plt.xlabel('Date')
plt.ylabel('Closing price')
plt.legend()
plt.show()
st.pyplot(fig2)