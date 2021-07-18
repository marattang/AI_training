import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import time

# 1. 데이터
# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000,) 흑백데이터이기 때문에 3차원
# print(x_test.shape, y_test.shape)   (10000, 28, 28) (10000,)

# 전처리
x_train = x_train.reshape(60000, 28, 28, 1)
# 데이터의 내용물과 순서가 바뀌면 안된다.
x_test = x_test.reshape(10000, 28, 28, 1)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
print('y_shape : ', y_train.shape)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
y_train = np.c_[y_train.toarray()]
y_test = encoder.fit_transform(y_test)
y_test = np.c_[y_test.toarray()]

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델링
model = Sequential()
model.add(Conv2D(filters=240, activation='relu', kernel_size=(2,2), padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(120, (2,2), activation='relu', padding='same'))          # (N, 9, 9, 20)
model.add(Conv2D(50, (2,2), activation='relu', padding='same'))          # (N, 9, 9, 20)
model.add(Conv2D(30, (2,2), padding='same', activation='relu'))             # (N, 8, 8, 30)
model.add(Conv2D(15, (1), activation='relu', padding='same'))                              # (N, 3, 3, 15)
model.add(Flatten())                                      # (N, 135)
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary

# 3. 컴파일, 훈련       metrics['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(mode='min', monitor='val_loss', patience=20)
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=500, validation_split=0.05, callbacks=[es])
# model.fit(x_train, y_train, epochs=50, batch_size=500, validation_split=0.1)
end = time.time() - start

print('걸린시간 : ', end)
# 4. 평가, 예측 predict X
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# acc로만 평가, earlystopping 사용
# batch=1000, epochs=500, validation_split=0.05, accuracy :  0.8989999890327454
# batch=1000, epochs=500, validation_split=0.1 , accuracy :  0.9006999731063843
# batch=500, epochs=500, validation_split=0.1 , accuracy :  0.9018999934196472
# EarlyStopping patience 15->20

# convolution node, layer 감소
# batch=500, epochs=500, validation_split=0.1 , accuracy :  0.8914999961853027 걸린시간 :  88.99784231185913

# Dnese layer 증가
# 니머지 하이퍼퍼라마터는 73번째 줄과 같음. accuracy :  0.8977000117301941 걸린시간 :  89.90077638626099
# batch=500, epochs=500, validation_split=0.15 , accuracy :  0.8934999704360962 걸린시간 :  88.99784231185913

# * early stopping 사용 x, epochs=50 => accuracy :  0.8939999938011169, accuracy :  0.8938999772071838
# Node의 수와 epochs의 수가 적었기에 과적합이 일어나지 않는 것같다.

# convolution layer에서 filter size (2,2) -> (1)로 감소, 나머지 하이퍼파라미터는 동일 : 별로 차이는 없어보인다.
# accuracy :  0.8877000212669373, accuracy :  0.891700029373169

# 모든 convolution layer에 padding 추가
# accuracy :  0.8985999822616577


#################
# 종합적으로 봤을 때 Convolution layer, node가 증가했을 경우, batch size가 줄었을 경우 acc가 증가하는 모습을 볼 수 있다.
# dense layer 증가, convolution layer증가시 훈련 데이터의 acc가 증가, 
# Early Stopping 사용x epochs 50 고정시
# accuracy :  0.9031000137329102 걸린시간 :  564.9791307449341
# Early Stopping 사용
# accuracy :  0.9034000039100647 걸린시간 :  302.1197626590729
# accuracy :  0.9075000286102295
