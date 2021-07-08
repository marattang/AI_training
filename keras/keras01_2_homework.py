import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6]

# 2.모델
model = Sequential()
model.add(Dense(1, input_dim =1))

# 3.컴파일
model.compile(loss="mse", optimizer='adam')
model.fit(x, y, epochs=2, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('6의 예측 값 : ', result)

# 완성한 뒤, 출력결과 스샷

