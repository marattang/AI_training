from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from tensorflow.keras.layers import concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# keras18_ansemble 사용하지 않고 모델링 r2 스코어 :  0.9999999335758925
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.array([range(1001, 1101)])
# y1 = np.array(range(1001, 1101)) [] 빼면 (100,)
y1 = np.transpose(y1)

print(x1.shape, x2.shape, y1.shape)

x1_train, x1_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=8, shuffle=True)

# 모델 구성
# 실습


# #2-1 모델1
input1 = Input(shape=(3,))
dense1 = Dense(55, activation='relu', name='dense1')(input1)
dense2 = Dense(32, activation='relu', name='dense2')(dense1)
dense3 = Dense(26, activation='relu', name='dense3')(dense2)
output1 = Dense(1)(dense3)

# #2-2 모델2
# input2 = Input(shape=(3,))
# dense11 = Dense(45, activation='relu', name='dense11')(input2)
# dense12 = Dense(28, activation='relu', name='dense12')(dense11)
# dense13 = Dense(20, activation='relu', name='dense13')(dense12)
# dense14 = Dense(10, activation='relu', name='dense14')(dense13)
# output2 = Dense(1)(dense14)

# last_output = Dense(1)(merge1)

model = Model(inputs=input1, outputs=output1)
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae']) # metrics=['mae','mse]
# 매트릭스를 보면 list로 받아들이고 있기 때문에 2개 이상을 쓰는 것도 가능하다.
model.fit(x1, y1, epochs=380, batch_size=25, verbose=1, validation_split=0.1)

# # 4. 평가, 예측
y_predict = model.predict(x1_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)