import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

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
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=10)

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 
model = Sequential()
model.add(Dense(256, input_shape=(13,)))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(3, activation='softmax'))

#
es = EarlyStopping(monitor='val_loss', mode='min', patience=15)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist = model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.1, callbacks=[es])
hist = model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.1)

plt.plot(hist.history['loss'])      # x: epoch, y: hist.history['loss']
plt.plot(hist.history['val_loss'])

plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.show()
#
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# QuantileTransformer - accuracy :  0.9259259104728699
# MaxAbsScaler - accuracy :  0.9259259104728699
# MinMaxScaler - accuracy :  0.9629629850387573
# RobustScaler - accuracy :  0.9814814925193787
# StandardScaler - accuracy :  0.9814814925193787
# PowerTransformer - accuracy :  0.9814814925193787

# 0.99까지 acc올리기