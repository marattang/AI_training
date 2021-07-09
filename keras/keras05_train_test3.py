from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(100)) # 0 ~ 99
y = np.array(range(1, 101)) # 1 ~ 100

# data = random.shuffle(np.stack((x,y), axis=1))

x_train, x_test, y_train, y_test = train_test_split(x, y,
         test_size=0.2, shuffle=True, random_state=66)

print(x_test)
print(y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1)) #첫번째 hidden layer가 노드가 5개라면, input은 1개.
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 컴퓨터가 이해하도록 컴파일

model.fit(x_train, y_train, epochs=1, batch_size=1) # 훈련시키기

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([11])
print('11의 예측값 : ', y_predict)

# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()