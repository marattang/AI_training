#1. R2를 음수가 아닌 0.5 이하로 만들어라.
#2. 데이터 건들지 마
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4. batch_size = 1
#5. epochs는 100 이상
#6. hidden layer node는 10개 이상 1000개 이하
#7. train 70%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
x = np.array(range(100)) # 0 ~ 99
y = np.array(range(1, 101)) # 1 ~ 100

# data = random.shuffle(np.stack((x,y), axis=1))

x_train, x_test, y_train, y_test = train_test_split(x, y,
         test_size=0.3, shuffle=True)

print(x_test)
print(y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1)) #첫번째 hidden layer가 노드가 5개라면, input은 1개.
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='kdl', optimizer='adam') # 컴퓨터가 이해하도록 컴파일

model.fit(x_train, y_train, epochs=100, batch_size=1) # 훈련시키기

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)

print('100의 예측값 : ', y_predict)
print('r2스코어 : ', r2)


# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()