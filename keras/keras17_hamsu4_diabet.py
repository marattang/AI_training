import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import r2_score
from keras import optimizers
import matplotlib.pyplot as plt

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)
print(x)
print(y)
print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))

# 

# batch size와 epoch를 둘다 늘릴 경우, 일반화된 데이터의 과적합으로 인해 val_loss가 하락하는 것
#2. 모델구성

# model = Sequential()
# model.add(Dense(55, input_shape=(10, ), activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(25, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(7, activation='relu'))
# model.add(Dense(1))
# 함수형 모델이 더 r2값이 적게 나오고 있다.
input1 = Input(shape=(10,))
dense1 = Dense(55)(input1)
dense2 = Dropout(0.2)(dense1)
dense3 = Dense(25)(dense2)
dense4 = Dropout(0.1)(dense3)
dense5 = Dense(12)(dense4)
dense6 = Dense(7)(dense5)
dense7 = Dense(3)(dense6)
output1 = Dense(1)(dense7)

model = Model(outputs=output1, inputs=input1)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=9, train_size=0.7)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, validation_split=0.1, batch_size=25)
# batch size, epochs 조정

#4. 평가, 예측
# mse, R2
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


# 과제 2
# r2 score :  0.6219390842668302