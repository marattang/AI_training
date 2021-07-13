import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from tensorflow.keras.losses import sparse_categorical_crossentropy

# 1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

# 2.모델
# 2-1. sequential  0.8100000429152601
# model = Sequential()
# model.add(Dense(55, input_dim =1))
# model.add(Dense(25))
# model.add(Dense(12))
# model.add(Dense(7))
# model.add(Dense(3))
# model.add(Dense(1))

# 2-2. 함수형 0.8100002431868347
input1 = Input(shape=(1,))
dense1 = Dense(55)(input1)
dense2 = Dense(25)(dense1)
dense3 = Dense(12)(dense2)
dense4 = Dense(7)(dense3)
dense5 = Dense(3)(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)

# 3.컴파일
# model.compile(loss="mse", optimizer='adam', loss_weights=1)
model.compile(loss="mse", optimizer='adam', loss_weights=1)
model.fit(x, y, epochs=2000, batch_size=32)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('6의 예측 값 : ', y_predict)
r2 = r2_score(y, y_predict)
print('r2 score : ', r2)

# 과제 2
# R2를 0.9 올려라!!!
