# 과제3 loss, r2출력
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gen_data_flow_ops import resource_accumulator_apply_gradient_eager_fallback
import numpy as np
# 데이터를 0 ~ 1사이의 값을 가진 데이터로 전처리를 하고 모델을 돌렸더니 정확도가 올라간다.
datasets = load_boston()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x)) # 100부터 시작한다고 하면 기준점이 명확하지 않음.
# 컬럼이 13개이기 때문에 컬럼마다 최소값 최대값이 다르기 때문에 별도로 따로 따로 해주는 것이 맞음.
# 
# 데이터 전처리
x = x/711
x = x/np.max(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, train_size=0.7, shuffle=True)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam", loss_weights=1)
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)
print('6의 예측 값 : ', y_predict)

r2 = r2_score(y, y_predict)
print('r2 score : ', r2)

# B = 흑인의 비율
# input 13, output 1(506)

# 완료하시오.