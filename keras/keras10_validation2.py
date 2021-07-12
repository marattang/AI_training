from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

# 잘라서 만들기
x_train = x[:7]
y_train = y[:7]

x_test = x[7:10]
y_test = y[7:10]

x_val = x[10:13]
y_val = y[10:13]

# x_train = np.array([1,2,3,4,5,6,7]) # 훈련, 공부하는 거 
# y_train = np.array([1,2,3,4,5,6,7])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
# x_test = np.array([8,9,10]) # 평가하는 거
# y_test = np.array([8,9,10])
# x_val = np.array([11,12,13])
# y_val = np.array([11,12,13])

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1)) #첫번째 hidden layer가 노드가 5개라면, input은 1개.
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

# #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 컴퓨터가 이해하도록 컴파일

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val)) # 훈련시키는 fit에서 검증을 하게 된다.

# #4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([11])
print('11의 예측값 : ', y_predict)

# # plt.scatter(x, y)
# # plt.plot(x, y_predict, color='red')
# # plt.show()