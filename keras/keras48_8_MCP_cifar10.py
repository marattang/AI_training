# 완벽한 모델 구성import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.python.keras.layers.core import Dropout
import numpy as np
import time
# 1. 데이터
# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 전처리
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
print('y_shape : ', y_train.shape)
y_train = y_train.reshape(50000, 1)
y_test = y_test.reshape(10000, 1)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
y_train = np.c_[y_train.toarray()]
y_test = encoder.fit_transform(y_test)
y_test = np.c_[y_test.toarray()]

# 2. 모델링
# CNN
model = Sequential()
model.add(Conv2D(filters=100, activation='relu', kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(100, (2,2), activation='relu', padding='same'))
model.add(Conv2D(100, (2,2), activation='relu', padding='same'))
model.add(Conv2D(100, (2,2), activation='relu', padding='same'))
model.add(Conv2D(100, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(100, (3,3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# RNN
# model = Sequential()
# model.add(LSTM(8, activation='relu', input_shape=(32*32, 3)))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련       metrics['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(mode='min', monitor='val_loss', patience=5)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', filepath='./_save/ModelCheckPoint/keras48_8_MCP.hdf', save_best_only=True)
start = time.time()
# model.fit(x_train, y_train, epochs=500, batch_size=512, validation_split=0.05, callbacks=[es, cp])
end = time.time() - start

print("걸린시간 : ", end)
# model.save('./_save/ModelCheckPoint/keras48_8_model.h5')
# model =load_model('./_save/ModelCheckPoint/keras48_8_model.h5')
model = load_model('./_save/ModelCheckPoint/keras48_8_MCP.hdf')

# 4. 평가, 예측 predict X
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# acc로만 평가
# batch_size 500 validation_split=0.05
# accuracy :  0.5238999724388123, 걸린시간 :  217.13401818275452
# accuracy :  0.5625    걸린시간 :  201.15319895744324

# 각 convolution node filter 증가
# accuracy :  0.6050000190734863, 걸린시간 :  352.5831444263458

# convolution node filter 증가, Maxpoolding 1번 사용
# accuracy :  0.6455000042915344, 걸린시간 :  123.45157933235168
# Maxpoolding 2번 사용
# accuracy :  0.6873000264167786, 걸린시간 :  66.12268400192261

# validation split 감소
# accuracy :  0.7071999907493591, 걸린시간 :  64.77738237380981

# bath size 감소
# accuracy :  0.7117000222206116, 걸린시간 :  81.92319917678833

# RNN
# 걸린시간 :  247.77784490585327
# accuracy :  0.10000000149011612

# model
# loss :  1.180648684501648
# accuracy :  0.6843000054359436

# load model
# loss :  1.180648922920227
# accuracy :  0.6843000054359436

# check point
# loss :  0.927131712436676
# accuracy :  0.685699999332428
