import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 1. 데이터 2x10인걸 10행 2열로 바꿔야함.
x = np.array([range(10), range(21, 31),
               range(201, 211)])
# 행무시, 열 우선 input layer
x = np.transpose(x)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               [1, 1.1, 1.2, 1.3, 1.4, 1.5, 
                1.6, 1.5, 1.4, 1.3],
                [10,9,8,7,6,5,4,3,2,1]])

y = np.transpose(y)
#  완성하시오

# 2. 모델 구성
model1 = Sequential()
model2 = Sequential()
model3 = Sequential()
# 역피라미드형 은닉화 모델
model1.add(Dense(9, input_dim=3))
model1.add(Dense(9))
model1.add(Dense(3))
model1.add(Dense(3))

# 피라미드형 은닉화 모델
model2.add(Dense(3, input_dim=3))
model2.add(Dense(9))
model2.add(Dense(9))
model2.add(Dense(3))

# 모래시계형 은닉화 모델
model3.add(Dense(9, input_dim=3))
model3.add(Dense(3))
model3.add(Dense(9))
model3.add(Dense(3))

# 3. 컴파일 훈련
model1.compile(loss='mse', optimizer='adam')
model2.compile(loss='mse', optimizer='adam')
model3.compile(loss='mse', optimizer='adam')

model1.fit(x, y, epochs=3000, batch_size=1)
model2.fit(x, y, epochs=3000, batch_size=1)
model3.fit(x, y, epochs=3000, batch_size=1)

# 4. 평가, 예측
x_pred = np.array([[0, 21, 201]]) # 3열이 나와야 하기 때문에 []
# x_pred2 = np.array([10, 1.3, 1]) 3행이 되기 때문에 안됨. 묶어줘야함.

# model1 역피라미드형
loss1 = model1.evaluate(x, y)
result1 = model1.predict(x_pred)

# model2 피라미드형
loss2 = model2.evaluate(x, y)
result2 = model2.predict(x_pred)

# model3 모래시계형
loss3 = model3.evaluate(x, y)
result3 = model3.predict(x_pred)


print('loss1 : ', loss1)
print('10, 1.3, 1의 예측값 : ', result1)


print('loss1 : ', loss2)
print('10, 1.3, 1의 예측값 : ', result2)


print('loss1 : ', loss3)
print('10, 1.3, 1의 예측값 : ', result3)
