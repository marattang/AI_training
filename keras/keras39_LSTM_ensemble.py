import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input, GRU, concatenate

# 1. 데이터
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
x2 = np.array([[10,20,30], [20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = np.array([55,65,75])
x2_predict = np.array([65,75,85])

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

x1_predict = x1_predict.reshape(1, x1_predict.shape[0], 1)
x2_predict = x2_predict.reshape(1, x2_predict.shape[0], 1)
# 3. 모델

input1 = Input(shape=(3,1))
dense1 = LSTM(units=32, activation='relu', input_shape=(3, 1))(input1)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(16, activation='relu')(dense2)
dense4 = Dense(16, activation='relu')(dense3)

input11 = Input(shape=(3,1))
dense11 = LSTM(units=32, activation='relu', input_shape=(3, 1))(input11)
dense12 = Dense(32, activation='relu')(dense11)
dense13 = Dense(16, activation='relu')(dense12)
dense14 = Dense(16, activation='relu')(dense13)

merge1 = concatenate([dense14, dense4])
merge2 = Dense(4, activation='relu')(merge1)
output = Dense(1, activation='relu')(merge2)

model = Model(inputs=[input1, input11], outputs=output)

model.compile(loss='mse', optimizer='adam')
model.fit([x1,x2], y, batch_size=1, epochs=250) 

# # 4. 평가, 예측
result = model.predict([x1_predict, x2_predict])
print(result)