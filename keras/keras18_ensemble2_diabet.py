import numpy as np
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
# 1. 데이터
dataset = load_diabetes()
# 딥러닝을 돌릴 때 함부로 변수를 줄여서 모델을 돌리면 정확도가 떨어지는 것 같다.
# print(dataset.feature_names)
# print(dataset.DESCR)['age':나이/연속, 'sex':성별/이산변수, 'bmi':체질량지수, 'bp':평균 혈압, 's1':T-세포수, 
                    #  's2':저밀도 지단백, 's3':고밀도 지단백, 's4':감상선 자극호르몬, 's5':라모트리진, 's6':혈당 수치]
x1 = dataset.data[:,[0,1,2]]
x2 = dataset.data[:,3:]
y = dataset.target

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, random_state=66, test_size=0.2, shuffle=True)

# 2-1. 앙상블 모델 r2 최고값 : 0.09~
# input1 = Input(shape=(3,))
# dense1 = Dense(64)(input1)
# dense2 = Dense(32)(dense1)
# dense3 = Dense(16)(dense2)
# dense4 = Dense(8)(dense3)
# dense5 = Dense(4)(dense4)
# output1 = Dense(1)(dense5)

# input2 = Input(shape=(7,))
# dense11 = Dense(64)(input2)
# dense12 = Dense(32)(dense11)
# dense13 = Dense(16)(dense12)
# dense14 = Dense(8)(dense13)
# dense15 = Dense(4)(dense14)
# output2 = Dense(1)(dense15)

# merge1 = concatenate([output1, output2])
# merge2 = Dense(24)(merge1)
# merge3 = Dense(15)(merge2)
# last_output = Dense(1)(merge3)

# model = Model(inputs=[input1, input2], outputs=last_output)

# 2-2. x2 모델 r2 테스트 = r2 = 0.3~
# input2 = Input(shape=(7,))
# dense11 = Dense(64)(input2)
# dense12 = Dense(32)(dense11)
# dense13 = Dense(16)(dense12)
# dense14 = Dense(8)(dense13)
# dense15 = Dense(4)(dense14)
# output2 = Dense(1)(dense15)

# 2-2. x1 모델 r2 테스트 = r2 = 0.3~
input1 = Input(shape=(3,))
dense1 = Dense(64)(input1)
dense2 = Dense(32)(dense1)
dense3 = Dense(16)(dense2)
dense4 = Dense(8)(dense3)
dense5 = Dense(4)(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics='mae')
# model.fit([x1_train, x2_train], y_train, batch_size = 212, epochs=500, validation_split=0.1)
# model.fit( x2_train, y_train, batch_size = 212, epochs=500, validation_split=0.15)
model.fit( x1_train, y_train, batch_size = 212, epochs=500, validation_split=0.15)

# 4. 실행, 예측
# result = model.evaluate([x1_test, x2_test], y_test) # evaluate는 loss와 metrics를 출력한다.
# result = model.evaluate([x1_test, x2_test], y_test)
# print('result : ', result)
# y_predict = model.predict([x1_test, x2_test])
# r2 = r2_score(y_test, y_predict)
# print('r2 : ', r2)

# result = model.evaluate(x2_test, y_test)
# print('result : ', result)
# y_predict = model.predict(x2_test)
# r2 = r2_score(y_test, y_predict)
# print('r2 : ', r2)

result = model.evaluate(x1_test, y_test)
print('result : ', result)
y_predict = model.predict(x1_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)