import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, LSTM, Conv1D
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score


# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape) #(442, 10)
print(y.shape) #(442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=10)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(309, 10, 1)
x_test = x_test.reshape(133, 10, 1)

# 2. 모델
# model = Sequential()
# model.add(LSTM(units=128, activation='relu', input_shape=(10, 1)))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))
model = Sequential()
model.add(Conv1D(filters=8, kernel_size=1, activation='relu', input_shape=(10, 1)))
# model.add(Conv1D(16, 1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 학습
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(mode='auto', monitor='val_loss', patience=15)
hist = model.fit(x_train, y_train, callbacks=[es], epochs=200, batch_size=1, validation_split=0.1)
# hist = model.fit(x_train, y_train, epochs=500, batch_size=5, validation_split=0.1)

# 4. 예측
# 1

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

# DNN
# R2 Score: 0.6 정도
# CNN
# r2 score :  0.4360472573203681
# RNN
# r2 score :  0.29323072315194054
# r2 score :  0.1988015803993728

# Conv1d 2번
# r2 score :  0.28487615204602945

# Conv1d 1번
# r2 score :  0.4591651397535842
# filters 감소 : 