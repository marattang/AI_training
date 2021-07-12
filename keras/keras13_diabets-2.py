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

#2. 모델구성

model = Sequential()
model.add(Dense(128, input_shape=(10, ), activation='relu'))
model.add(Dense(25, activation='relu')) # 활성화함수, 모든 레이어에 존재한다. 기본적으로 relu하면 통상 85%이상 좋은 결과가 나온다.
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1)) # 파라미터 튜닝시 epochs, batch size, node 개수, layer의 깊이 마지막 레이어에서는 절대 activation 쓰면 안된다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.1)

#4. 평가, 예측
# mse, R2

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)
print('예측값 : ', y_predict)

r2 = r2_score(y, y_predict)
print('r2 score : ', r2)