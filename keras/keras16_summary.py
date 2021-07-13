import numpy as np
from numpy.core.fromnumeric import transpose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101),
            range(100), range(401, 501)])

x = np.transpose(x)
print(x.shape)  # (100, 5)
y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)
print(y.shape)  # (100, 2)

#2. 모델
model = Sequential()
model.add(Dense(3, input_shape=(5,)))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(2))

model.summary()

#3. 컴파일, 훈련
#4. 평가, 예측
