from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from tensorflow.keras.layers import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.array([range(1001, 1101)])
y2 = np.array(range(1901, 2001)) # transpose 할 필요x
y1 = np.transpose(y1)
print(x1.shape, x2.shape, y1.shape)

# x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y1, test_size=0.2, random_state=8, shuffle=True)
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, y1, y2, test_size=0.3, random_state=8, shuffle=True)

print(x1_train.shape, x2_train.shape, y1_train.shape, y2_train.shape)
print(x1_test.shape, x2_test.shape, y1_test.shape, y2_test.shape)

# 모델 구성
# 실습


# #2-1 모델1
input1 = Input(shape=(3,))
dense1 = Dense(55, activation='relu', name='dense1')(input1)
dense2 = Dense(32, activation='relu', name='dense2')(dense1)
dense3 = Dense(26, activation='relu', name='dense3')(dense2)
output1 = Dense(18)(dense3)

# #2-2 모델2
input2 = Input(shape=(3,))
dense11 = Dense(45, activation='relu', name='dense11')(input2)
dense12 = Dense(28, activation='relu', name='dense12')(dense11)
dense13 = Dense(20, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(7)(dense14)

merge1 = concatenate([output1, output2]) # 첫번째 모델의 가장 마지막 부분, 두번째 모델의 가장 마지막 부분 병합.
# 과제 4. Concentenate로 코딩
merge1 = Concatenate(axis=1)([output1, output2])


merge2 = Dense(24)(merge1)
merge3 = Dense(15, activation='relu')(merge2)
# last_output = Dense(1)(merge3)
output21 = Dense(7)(merge3)
last_output1 = Dense(1, name='outputdense1')(output21)

output22 = Dense(8)(merge3)
last_output2 = Dense(1, name='outputdense2')(output22)

model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae']) # metrics=['mae','mse]
# 매트릭스를 보면 list로 받아들이고 있기 때문에 2개 이상을 쓰는 것도 가능하다.
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=8, verbose=1, validation_split=0.1)

# # 4. 평가, 예측
result = model.evaluate([x1_test, x2_test], [y1_test, y2_test]) # evaluate는 loss와 metrics를 출력한다.
print('result : ', result)
y_predict = model.predict([x1_test, x2_test])

print('loss : ', result[0])
print('metrics["mae"] : ', result[1])

#r2 스코어 :  0.9914715240776343