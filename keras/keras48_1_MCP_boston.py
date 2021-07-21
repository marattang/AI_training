
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.python.keras.engine.training import Model
# 데이터를 0 ~ 1사이의 값을 가진 데이터로 전처리를 하고 모델을 돌렸더니 정확도가 올라간다.
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=20, train_size=0.7, shuffle=True)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

es = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', filepath='./_save/ModelCheckPoint/keras48_1_MCP.hdf', save_best_only=True)
model.compile(loss="mse", optimizer="adam", loss_weights=1)
model.fit(x_train, y_train, epochs=350, batch_size=32, validation_split=0.1, callbacks=[es, cp])

# model.save('./_save/ModelCheckPoint/keras48_1_model.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_1_model.h5')
model = load_model('./_save/ModelCheckPoint/keras48_1_MCP.hdf')

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

# 저장 실행
# r2 score :  0.8603100281026542

# model
# r2 score :  0.8603100281026542

# check point