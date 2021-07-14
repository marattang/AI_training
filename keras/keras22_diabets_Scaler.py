from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))

# case1 연산이 너무 적음 5,4,3,2,1
# case2 노드가 많다가 갑자기 줄어듬. 400, 243, 3, 1
# case3. 1000, 5883, 840, 1233, 1102, 8335 통상적으로 역삼각형 형태가 가장 많음.
# 한 레이어당 지나치게 많은 노드의 수를 배정할 경우 과적합이 일어난다.

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=9, train_size=0.7)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(55, input_shape=(10, ), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(12, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=220, validation_split=0.1, batch_size=45)


#4. 평가, 예측
# mse, R2
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


# 과제 2
# 데이터 전처리 하지 않았을 때,
# r2 score 최고값:  0.6219390842668302
# 보통 0.61~ 0.616

# MaxAbsScaler
# r2 score :  0.6202291136572824
# 보통 0.61 ~ 0.618

# RobustScaler
# r2 score :  0.5983461393450015
# 보통 0.58 ~ 0.59

# QuantileTransformer
# r2 score :  0.5708573913980365
# 보통 0.56~0.57

# PowerTransformer
# r2 score :  0.585200727625568
# 보통 0.56 ~ 0.58