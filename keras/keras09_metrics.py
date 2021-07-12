import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time

# 1. 데이터 2x10인걸 10행 2열로 바꿔야함.
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               [1, 1.1, 1.2, 1.3, 1.4, 1.5, 
                1.6, 1.5, 1.4, 1.3],
                [10,9,8,7,6,5,4,3,2,1]])
# 행무시, 열 우선 input layer
x = np.transpose(x)
print(np.shape(x))
y = np.array( [11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#  완성하시오

# 2. 모델 구성
model = Sequential()
model.add(Dense(9, input_dim=3))
model.add(Dense(9))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

start = time.time()
model.fit(x, y, epochs=1000, batch_size=10, verbose=1)
end = time.time() - start
print('걸린시간 : ', end)

# 4. 평가, 예측

loss = model.evaluate(x, y)
result = model.predict(x)

print('loss : ', loss)
print('의 예측값 : ', result)
print(x.shape)