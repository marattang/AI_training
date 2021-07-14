# 과제3 loss, r2출력
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gen_data_flow_ops import resource_accumulator_apply_gradient_eager_fallback
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# 데이터를 0 ~ 1사이의 값을 가진 데이터로 전처리를 하고 모델을 돌렸더니 정확도가 올라간다.
datasets = load_boston()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x)) # 100부터 시작한다고 하면 기준점이 명확하지 않음.
# 컬럼이 13개이기 때문에 컬럼마다 최소값 최대값이 다르기 때문에 별도로 따로 따로 해주는 것이 맞음.
# 



# 데이터 전처리
# x = (x - np.min(x)) / (np.max(x) - np.max(x)) # (x-min)/(max-min) 컬럼별로 최대 최소를 잡아야하기 때문에 정확도가 낮게 나옴.
# train, test 전체에 대해 scaling하면 train데이터에 과적합된다. 데이터의 범위가 다 잡혀있어서 훈련 데이터와 테스트 데이터를 분리해놨담녀
# 훈련데이터로 훈련을 시켜야 하는데, 테스트의 값이 훈련데이터에 반영되면 안됨
# 평가데이터는 훈련데이터와 범위가 약간 틀어질 수도 있다. 이것도 훈련 데이터에 반영이 되면 안되기 때문에
scaler = MinMaxScaler()
scaler.fit(x) # 훈련시키기 메모리상 실행만 됨
x_scale = scaler.transform(x) # 변환시키기. 바꾼걸 배치시켜줌

print(x_scale[:10])
print(np.min(x_scale), np.max(x_scale))

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, random_state=66, train_size=0.7, shuffle=True)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam", loss_weights=1)
model.fit(x_train, y_train, epochs=250, batch_size=32, validation_split=0.1)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

# r2 score :  0.9101474776933229
# minmaxscalar
# r2 = 0.8~
# not
# r2 = 0.7~