# 과제3 loss, r2출력
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data
y = datasets.target

# model = Sequential()
# model.add(Dense(9, input_dim=13))
# model.add(Dense(8))
# model.add(Dense(1))

input1 = Input(shape=(13,))
dense1 = Dense(9)(input1)
dense2 = Dense(8)(dense1)
output1 = Dense(1)(dense2)

model = Model(inputs=input1, outputs=output1)

model.compile(loss="mse", optimizer="adam", loss_weights=1)
model.fit(x, y, epochs=500, batch_size=1)

loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('6의 예측 값 : ', y_predict)

r2 = r2_score(y, y_predict)
print('r2 score : ', r2)

# B = 흑인의 비율
# input 13, output 1(506)

# 완료하시오.