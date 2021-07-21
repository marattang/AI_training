from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error







# 실습
# 1~100까지의 데이터를

#     x            y
# 1,2,3,4,5        6
# ...
# 95,96,97,98,99   100

import numpy as np

x_data = np.array(range(1, 101))
x_predict = np.array(range(96, 105))

# 96, 97, 98, 99, 100     ?
# ...
# 101, 102, 103, 104, 105 ?

size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset_x = split_x(x_data, size)
x_predict = split_x(x_predict, size-1)

x = dataset_x[:, :5]
y = dataset_x[:, 5]

# print("x : \n", x)
# print("y : \n", y)
# print("x_predict", dataset_x_predict)
# 이해하기

# print("x shape", x.shape)
# print("x predict shape", dataset_x_predict.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=10)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1])
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1])
# x_predict = x_predict.reshape(5,5)

print('x_pre : ', x_predict)
print('x_pre_shape : ', x_predict.shape)
print('x_shape : ', x_train.shape)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=5))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

# 
es = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1, callbacks=[es], validation_split=0.1)

# 
result = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print('r2 score : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

print("rmse스코어 : ", rmse)

# RNN
# r2 score :  0.9998582743578431
# rmse스코어 :  0.31132974011585923

# DNN
# r2 score :  0.9999765150714409
# rmse스코어 :  0.1260984567642131

# predict
pre = model.predict(x_predict)
print("predict : ", pre)

# predict :  [[48555.75 ]
#  [49025.43 ]
#  [49494.766]
#  [49963.805]
#  [50432.547]]