from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from tensorflow.keras.layers import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from tensorflow.python.keras.saving.save import load_model

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.array([range(1001, 1101)])
# y1 = np.array(range(1001, 1101)) [] 빼면 (100,)
y1 = np.transpose(y1)

# print(x1.shape, x2.shape, y1.shape)

# x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y1, test_size=0.2, random_state=8, shuffle=True)
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y1, random_state=8, shuffle=True)

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
last_output = Dense(1)(merge3)

# last_output = Dense(1)(merge1)

model = Model(inputs=[input1, input2], outputs=last_output)
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae']) # metrics=['mae','mse]
# 매트릭스를 보면 list로 받아들이고 있기 때문에 2개 이상을 쓰는 것도 가능하다.

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1,
                    restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                        filepath='./_save/ModelCheckPoint/keras49_mcp.h5')

model.fit([x1_train, x2_train], y_train, epochs=400, batch_size=25, verbose=1, validation_split=0.1,
            callbacks=[es, mcp])

model.save('./_save/ModelCheckPoint/keras49_model_save.h5')

print('===============1. 기본 출력==============')

# # 4. 평가, 예측
result = model.evaluate([x1_test, x2_test], y_test) # evaluate는 loss와 metrics를 출력한다.
print('result : ', result[0])

y_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)


#r2 스코어 :  0.9914715240776343 -> 0.9997684219501827
# loss 소수점단위까지 낮추기 -> 0.20147289335727692

print('===============2. load_model ==============')
model2 = load_model('./_save/ModelCheckPoint/keras49_model_save.h5')

result = model2.evaluate([x1_test, x2_test], y_test) # evaluate는 loss와 metrics를 출력한다.
print('result : ', result[0])

y_predict = model2.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

print('===============3. Model check ==============')
model3 = load_model('./_save/ModelCheckPoint/keras49_mcp.h5')

result = model3.evaluate([x1_test, x2_test], y_test) # evaluate는 loss와 metrics를 출력한다.
print('loss : ', result[0])

y_predict = model3.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)
'''
* restore_best_Weight = False

===============1. 기본 출력==============
loss :  5.251635551452637
r2 스코어 :  0.9939636367879723
===============2. load_model ==============
loss :  5.251635551452637
r2 스코어 :  0.9939636367879723
===============3. Model check ==============
loss :  7.454799652099609
r2 스코어 :  0.991431264601391
'''

'''
* restore_best_Weight = True break지점의 weight를 저장한다는 얘기.

===============1. 기본 출력==============
result :  0.4238952398300171
r2 스코어 :  0.9995127640979729
===============2. load_model ==============
result :  0.4238952398300171
r2 스코어 :  0.9995127640979729
===============3. Model check ==============
result :  0.4238952398300171
r2 스코어 :  0.9995127640979729
'''