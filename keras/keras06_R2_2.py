import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from tensorflow.keras.losses import sparse_categorical_crossentropy

# 1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

# 2.모델
model = Sequential()
model.add(Dense(9, input_dim =1))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))



# 3.컴파일
# model.compile(loss="mse", optimizer='adam', loss_weights=1)
model.compile(loss=sparse_categorical_crossentropy, optimizer='adam', loss_weights=1)
model.fit(x, y, epochs=5000, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('6의 예측 값 : ', y_predict)
r2 = r2_score(y, y_predict)
print('r2 score : ', r2)

# 과제 2
# R2를 0.9 올려라!!!
