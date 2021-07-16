import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0)

# 1. 판다스 -> 넘파이
# 2. x, y를 분리
# 3. sklearn의 onehot??? 사용할것
# 3. y의 라벨을 확인 np.unique(y)
datasets = datasets.to_numpy()

# ./   : 현재 폴더
#  ../ : 상위 폴더

# print(datasets.shape) # (4898, 12)
# to_categorical은 label의 시작을 0으로 본다. 3부터 시작했기 때문에 0, 1, 2부터 라벨이 있다고 판단해 shape가 커진다.
x = datasets[:,:11]
y = datasets[:,[-1]]
# print(y)
encoder = OneHotEncoder()
y = encoder.fit_transform(y)
y = np.c_[y.toarray()]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, shuffle=True, random_state=2)

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print('&&&&&', np.unique(y))
# # # 
model = Sequential()
model.add(Dense(4200, input_shape=(11,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1400, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(7, activation='softmax'))

# 다중분류
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=500, validation_split=0.05)

# 모델링 하고
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
# 0.8 이상 완성
