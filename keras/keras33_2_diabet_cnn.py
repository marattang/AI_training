import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score


# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape) #(442, 10)
print(y.shape) #(442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=10)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(309, 10, 1, 1)
x_test = x_test.reshape(133, 10, 1, 1)

# 2. 모델
model = Sequential()
model.add(Conv2D(filters=128, activation='relu', kernel_size=(1,1), padding='valid', input_shape=(10, 1, 1)))
model.add(Conv2D(64, kernel_size=(1,1), activation='relu'))
model.add(Conv2D(64, kernel_size=(1,1), activation='relu'))
model.add(Conv2D(64, kernel_size=(1,1), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='softmax'))


# 3. 컴파일, 학습
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(mode='auto', monitor='val_loss', patience=15)
# hist = model.fit(x_train, y_train, callbacks=[es], epochs=500, batch_size=5, validation_split=0.1)
hist = model.fit(x_train, y_train, epochs=500, batch_size=5, validation_split=0.1)

# 4. 예측
# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()


# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.grid()
plt.title('mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['mae', 'val_mae'])
plt.show()

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)