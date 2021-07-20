import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(4, 3, 1) # (batch_size, timesteps, feature)

# 2. 모델구성
model = Sequential()
model.add(SimpleRNN(units=20, activation='relu', input_shape=(3, 1)))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


# (Input + bias) * output + output * output= (Input + bias + output) * output
# 이전 layer의 가중치가 반영되기 때문에 식이 완성된다.
# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, batch_size=1, epochs=1200) 

# # 4. 평가, 예측
# x_input = np.array([5, 6, 7]).reshape(1,3,1)
# result = model.predict(x_input)
# print(result)

# [[8.]]

model.summary()
