import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# 1.데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])


x_train, x_test, y_train, y_test = train_test_split(x, y,
         test_size=0.2, shuffle=True, random_state=66)


# 2.모델
model = Sequential()
model.add(Dense(9, input_dim =1))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))


# 3.컴파일
# model.compile(loss="mse", optimizer='adam', loss_weights=1)
model.compile(loss="categorical_crossentropy", optimizer='adam', loss_weights=25)
model.fit(x, y, epochs=5000, batch_size=25)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('6의 예측 값 : ', y_predict)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

print("rmse스코어 : ", rmse)
