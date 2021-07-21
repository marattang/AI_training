import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Flatten, LSTM, Conv1D
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import font_manager, rc
from tensorflow.python.keras.layers.core import Dropout
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 완성하시오
# acc 0.8 이상 만들것
dataset = load_wine()
x = dataset.data
y = dataset.target

print(dataset.DESCR)
print(dataset.feature_names)
print(np.unique(y))

y = to_categorical(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

print(x_train)
print(x_train.shape)

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape)
# print(x_test.shape)
x_train = x_train.reshape(124, 13, 1)
x_test = x_test.reshape(54, 13, 1)
# 

# model = Sequential()
# model.add(LSTM(units=128, activation='relu', input_shape=(13, 1)))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax'))

model = Sequential()
model.add(Conv1D(16, kernel_size=1, activation='relu', input_shape=(13, 1)))
model.add(Conv1D(8, 1))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

#
es = EarlyStopping(monitor='val_loss', mode='min', patience=15)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist = model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.1, callbacks=[es])
hist = model.fit(x_train, y_train, batch_size=1, epochs=70, validation_split=0.05)

# plt.plot(hist.history['loss'])      # x: epoch, y: hist.history['loss']
# plt.plot(hist.history['val_loss'])

# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
# plt.title('로스, 발로스')
# plt.show()

#
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# DNN
# QuantileTransformer - accuracy :  0.9259259104728699
# MaxAbsScaler - accuracy :  0.9259259104728699
# MinMaxScaler - accuracy :  0.9629629850387573
# RobustScaler - accuracy :  0.9814814925193787
# StandardScaler - accuracy :  0.9814814925193787
# PowerTransformer - accuracy :  0.9814814925193787

# CNN
# accuracy :  0.9814814925193787

# RNN 
# epochs 50 -> 70
# 하이퍼 파라미터 작업 후
# accuracy :  0.9444444179534912 -> accuracy :  1.0

# conv1d
# accuracy :  0.9814814925193787