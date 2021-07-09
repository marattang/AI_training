from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1)) #첫번째 hidden layer가 노드가 5개라면, input은 1개.
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 컴퓨터가 이해하도록 컴파일

model.fit(x_train, y_train, epochs=1000, batch_size=1) # 훈련시키기

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([11])
print('11의 예측값 : ', y_predict)

# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()