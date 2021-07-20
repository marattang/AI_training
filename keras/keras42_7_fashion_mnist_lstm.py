import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling1D, LSTM
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import time

from tensorflow.python.keras.layers.core import Dropout

# 1. 데이터
# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000,) 흑백데이터이기 때문에 3차원
# print(x_test.shape, y_test.shape)   (10000, 28, 28) (10000,)

# 전처리
x_train = x_train.reshape(60000, 28 * 28)
# 데이터의 내용물과 순서가 바뀌면 안된다.
x_test = x_test.reshape(10000, 28 * 28)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
print('y_shape : ', y_train.shape)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
y_train = np.c_[y_train.toarray()]
y_test = encoder.fit_transform(y_test)
y_test = np.c_[y_test.toarray()]

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28 * 28, 1)
x_test = x_test.reshape(10000, 28 * 28, 1)

# 2. 모델링
model = Sequential()

# CNN
# model.add(Conv2D(filters=240, activation='relu', kernel_size=(2,2), padding='same', input_shape=(28, 28, 1)))
# model.add(Conv2D(120, (2,2), activation='relu', padding='same'))          # (N, 9, 9, 20)
# model.add(Conv2D(50, (2,2), activation='relu', padding='same'))          # (N, 9, 9, 20)
# model.add(Conv2D(30, (2,2), padding='same', activation='relu'))             # (N, 8, 8, 30)
# model.add(Conv2D(15, (1), activation='relu', padding='same'))                              # (N, 3, 3, 15)
# model.add(Flatten())                                      # (N, 135)
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# DNN
# model.add(Dense(528, input_shape=(28 * 28, ), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# RNN
model.add(LSTM(16, input_shape=(28 * 28, 1), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary

# 3. 컴파일, 훈련       metrics['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(mode='min', monitor='val_loss', patience=5)
start = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=512, validation_split=0.05, callbacks=[es])
# model.fit(x_train, y_train, epochs=50, batch_size=500, validation_split=0.1)
end = time.time() - start

print('걸린시간 : ', end)
# 4. 평가, 예측 predict X
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# CNN의 경우
# accuracy :  0.9075000286102295
# DNN의 경우
# accuracy :  0.9014999866485596
# RNN
# Epochs 16회 전까지는 서서히 증가하다 val_loss가 nan값이 나오면서 accuracy 0.41 -> 0.1로 감소
# accuracy :  0.10000000149011612
