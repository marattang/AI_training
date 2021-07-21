import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Conv2D, Flatten, Dropout, LSTM, Conv1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

datasets = load_breast_cancer()
# 1. 데이터
# 데이터셋 정보 확인
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
y = to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66, shuffle=True)

print(x.shape, y.shape) # (input = 30, output = 1)

# 2. 모델
# input = Input(shape=(30,))
# dense1 = Dense(128)(input)
# dense2 = Dense(64)(dense1)
# dense3 = Dense(64)(dense2)
# dense4 = Dense(32)(dense3)
# dense5 = Dense(16)(dense4)
# output = Dense(1, activation='sigmoid')(dense5)
# 마지막 레이어의 activation은 linear, sigmoid로 간다. 0, 1의 값을 받고 싶으면 무조건 sigmoid사용. loss는 binary_crossentropy

scaler = QuantileTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(398, 30, 1)
x_test = x_test.reshape(171, 30, 1)

# 3. 컴파일, 훈련
# model = Sequential()
# model.add(LSTM(units=64, activation='relu', input_shape=(30, 1)))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(2, activation='softmax'))

model = Sequential()
model.add(Conv1D(filters=64, activation='relu', input_shape=(30, 1), kernel_size=1))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# metrics에 들어간 건 결과에 반영되지 않고 보여주기만 한다.
es = EarlyStopping(mode='min', monitor='val_loss', patience=5)

print(x_train.shape)
print(y_train.shape)
model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.1, callbacks=[es])

# 평가, 예측
loss = model.evaluate(x_test, y_test) # evaluate는 loss과 metrics도 반환한다. binary_crossentropy의 loss, accuracy의 loss
print('loss : ', loss[0])
print('accuracy : ', loss[1])
# CNN
# accuracy :  0.988304078578949
# RNN - LSTM
# accuracy :  0.9766082167625427