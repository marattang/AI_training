import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 1. 데이터 2x10인걸 10행 2열로 바꿔야함.
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               [1, 1.1, 1.2, 1.3, 1.4, 1.5, 
                1.6, 1.5, 1.4, 1.3]])
# 행무시, 열 우선 input layer
x = np.transpose(x)

y = np.array( [11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#  완성하시오

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.add(Dense(1, input_dim=2))

# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=5000, batch_size=50)

# 4. 평가, 예측
x_pred = np.array([[10, 1.3]]) # 2열이 나와야 하기 때문에 []
# x_pred2 = np.array([10, 1.3]) 2행이 되기 때문에 안됨. 묶어줘야함.

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('10, 1.3의 예측값 : ', result)

# 1. [1,2,3] (3,)
# 1-1. [[1], [2], [3]] (1, 3)
# 2. [[1,2,3]] (1, 3)
# 3. [[1,2],[3,4],[5,6]] (3, 2)
# 4. [[[1,2,3], [4,5,6]]] (1, 2, 3)
# 5. [[[1,2], [3,4], [5,6]]] (1, 3, 2)
# 6. [[[1], [2]], [[3], [4]]] (2, 2, 1)
