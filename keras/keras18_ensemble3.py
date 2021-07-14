from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from tensorflow.keras.layers import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x1 = np.array([range(100), range(301, 401), range(1, 101)])
# x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
# x2 = np.transpose(x2)
y1 = np.array([range(1001, 1101)])
y2 = np.array(range(1901, 2001)) # transpose 할 필요x
y1 = np.transpose(y1)
print(x1.shape, y1.shape)

# x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y1, test_size=0.2, random_state=8, shuffle=True)
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, test_size=0.3, random_state=8, shuffle=True)

# 모델 구성
# 실습


# #2-1 모델1
input1 = Input(shape=(3,))
dense1 = Dense(55, activation='relu', name='dense1')(input1)
dense2 = Dense(32, activation='relu', name='dense2')(dense1)
dense3 = Dense(26, activation='relu', name='dense3')(dense2)
output1 = Dense(18)(dense3)

# #2-2 모델2

modelense1 = Dense(24)(output1)
modelense2 = Dense(24)(modelense1)
modelense3 = Dense(24)(modelense2)
modelense4 = Dense(24)(modelense3)
output21 = Dense(7)(modelense4)


modelense11 = Dense(24)(output1)
modelense12 = Dense(24)(modelense11)
modelense13 = Dense(24)(modelense12)
modelense14 = Dense(24)(modelense13)
output22 = Dense(8)(modelense14)

# last_output = Dense(1)(merge3)
last_output1 = Dense(1, name='outputdense1')(output21)
last_output2 = Dense(1, name='outputdense2')(output22)

model = Model(inputs=input1, outputs=[last_output1, last_output2])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae']) # metrics=['mae','mse]
# 매트릭스를 보면 list로 받아들이고 있기 때문에 2개 이상을 쓰는 것도 가능하다.
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=8, verbose=1, validation_split=0.1)

# # 4. 평가, 예측
result = model.evaluate(x1_test, [y1_test, y2_test]) # evaluate는 loss와 metrics를 출력한다.
print('result : ', result)
y_predict = model.predict(x1_test)

print('loss : ', result[0])
print('metrics["mae"] : ', result[1])