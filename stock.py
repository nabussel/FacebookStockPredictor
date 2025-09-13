
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


#reading in the data and getting just the open values
fb_complete_data = pd.read_csv("fb_train.csv")
print(fb_complete_data.head())
fb_training_processed = fb_complete_data[['Open']].values#1257 values
print(fb_training_processed)

#scaling the data into an array that has values between 0 and 1, still 1257 values
scaler = MinMaxScaler(feature_range = (0,1))
fb_training_scaled = scaler.fit_transform(fb_training_processed)
print(fb_training_scaled)
print(len(fb_training_scaled))

#training features contain lists of 60 day sets of data
#labels contain the data for the 61st day 
fb_training_features = []# contains 1197 lists of 60 values
fb_training_labels = []#contains 1197 values
for i in range(60,len(fb_training_scaled)):
    fb_training_features.append(fb_training_scaled[i-60:i,0])
    fb_training_labels.append(fb_training_scaled[i,0])
print(len(fb_training_labels))


#converting our lists into numpy arrays to be used be other packages
X_train = np.array(fb_training_features)
Y_train = np.array(fb_training_labels)

#converts data into 3_D shape, 1 array of 1197 rows and 60 columns
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
print(X_train.shape)

#ReLu takes max(0,x) to introduce nonlinearity into algorithm
input_layer = tf.keras.layers.Input(shape = (X_train.shape[1],1))# 1 row of 60 columns
lstm1 = tf.keras.layers.LSTM(100, activation='relu', return_sequences=True)(input_layer)#100 nodes, returns full output sequence
do1 = tf.keras.layers.Dropout(0.2)(lstm1)#20 percent of the activation values turn to 0
lstm2 = tf.keras.layers.LSTM(100, activation='relu', return_sequences=True)(do1)
do2 = tf.keras.layers.Dropout(0.2)(lstm2)
lstm3 = tf.keras.layers.LSTM(100, activation='relu', return_sequences=True)(do2)
do3 = tf.keras.layers.Dropout(0.2)(lstm3)
lstm4 = tf.keras.layers.LSTM(100, activation='relu')(do3)
do4 = tf.keras.layers.Dropout(0.2)(lstm4)

output_layer = tf.keras.layers.Dense(1)(do4)#1 node
model = tf.keras.models.Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='mse')#adam quickly converges in changing weights and biases,unique to each parameter
#Mean Squared Error of all outputs compared to true output


print(X_train.shape) #1197 rows, 60 columns, 1 big array
print(Y_train.shape)#1197 rows
Y_train= Y_train.reshape(-1,1)
print(Y_train.shape)#1197 rows, 1 column

model_history = model.fit(X_train, Y_train, epochs=100,
verbose=1, batch_size = 32) #uses 32 samples in each epoch, 100 full iterations for 100 updates

fb_testing_complete_data = pd.read_csv("fb_test.csv")
fb_testing_processed = fb_testing_complete_data[['Open']].values
fb_all_data = pd.concat((fb_complete_data['Open'], fb_testing_complete_data['Open']), axis = 0)
#all the original training values added to the testing values

print(fb_complete_data['Open'])
print(fb_testing_complete_data['Open'])
print(fb_all_data)
print(fb_all_data.shape)
test_inputs = fb_all_data[len(fb_all_data)-len(fb_testing_complete_data)-60:].values
#gets last 80 values of the concatenated data, contains the testing data and the last 60 days of the training data
print(len(fb_all_data))
print(len(fb_testing_complete_data))
print(test_inputs)


test_inputs = test_inputs.reshape(-1,1) #80 rows 1 columns
print(test_inputs.shape)
test_inputs = scaler.transform(test_inputs)# all numbers match their normal distribution z-score
print(test_inputs)

fb_test_features = []
for i in range(60,80):#creates sets of 60 data values
    fb_test_features.append(test_inputs[i-60:i, 0])

X_test = np.array(fb_test_features)#20 rows and 60 columns
print(X_test.shape)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.
shape[1], 1))#makes 3d shape for predict
print(X_test.shape)

y_pred = model.predict(X_test) 
y_pred = scaler.inverse_transform(y_pred)#gets us back our original y value scale to plot

plt.figure(figsize=(8,6))
plt.plot(fb_testing_processed, color='red', label='Actual Facebook Stock Price')
plt.plot(y_pred , color='green', label='Predicted Facebook Stock Price')
plt.title('Facebook Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show() 






