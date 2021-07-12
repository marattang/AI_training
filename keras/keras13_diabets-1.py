import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

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

# 반장 연산이 너무 적음 5,4,3,2,1
# 인정 노드가 많다가 갑자기 줄어듬. 400, 243, 3, 1
# 형준 1000, 5883, 840, 1233, 1102, 8335 통상적으로 역삼각형 형태가 가장 많음.
model = Sequential()
model.add(Dense(400, input_shape=(10, ), activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(1))

#2. 모델구성
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.5)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=600, batch_size=212, validation_split=0.2)

#4. 평가, 예측
# mse, R2

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('예측값 : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)
# 과제 2
# 0.62 까지 올릴 것!