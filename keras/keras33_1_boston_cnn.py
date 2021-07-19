
import numpy as np
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dropout

# 데이터를 0 ~ 1사이의 값을 가진 데이터로 전처리를 하고 모델을 돌렸더니 정확도가 올라간다.
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=20, train_size=0.7, shuffle=True)

scaler = RobustScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(354, 13, 1, 1)
x_test = x_test.reshape(152, 13, 1, 1)

print(x.shape) #(506, 13)
print(y.shape) #(506,)

model = Sequential()
# model.add(Dense(128, activation='relu', input_dim=13))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Conv2D(filters=256, activation='relu', kernel_size=(1,1), padding='valid', input_shape=(1, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=128, activation='relu', kernel_size=(1,1)))
model.add(Conv2D(filters=64, activation='relu', kernel_size=(1,1)))
model.add(Conv2D(filters=32, activation='relu', kernel_size=(1,1)))
model.add(Conv2D(filters=16, activation='relu', kernel_size=(1,1)))
model.add(GlobalAveragePooling2D())
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam", loss_weights=1, metrics=['mae'])
es = EarlyStopping(mode='auto', monitor='val_loss', patience=15)
hist = model.fit(x_train, y_train, epochs=1200, batch_size=5, validation_split=0.05, callbacks=[es])

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

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
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

# RobustScaler
# DNN
# r2 score :  0.8773921354087534
# 보통 0.82 ~ 0.87
# CNN
# r2 score :  0.38696311110913495
# r2 score :  0.34668385029217896