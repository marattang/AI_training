# 과제3 loss, r2출력
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 데이터를 0 ~ 1사이의 값을 가진 데이터로 전처리를 하고 모델을 돌렸더니 정확도가 올라간다.
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, train_size=0.7, shuffle=True)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam", loss_weights=1)

# 
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es]) # callback에는 earlystopping말고 다른 게 들어갈 수 있음
# model.fit(x_train, y_train, epochs=350, batch_size=32, validation_split=0.1)
print(hist.history.keys()) # dict_keys(['loss', 'val_loss'])
print('==========loss=========') # [568.2015991210938, 507.7735290527344, 355.2822570800781]
print(hist.history['loss'])
print('==========val loss=========')
print(hist.history['val_loss']) # [611.8585205078125, 492.8374328613281, 231.34072875976562]

plt.plot(hist.history['loss'])      # x: epoch, y: hist.history['loss']
plt.plot(hist.history['val_loss'])

plt.title('로스, 발로스')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(['train loss', 'val_loss'])
plt.show()

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

# r2 score :  0.8696841262930644