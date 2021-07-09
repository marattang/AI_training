from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import random

#1. 데이터
x = np.array(range(100)) # 0 ~ 99
y = np.array(range(1, 101)) # 1 ~ 100

data = np.random.shuffle(np.stack((x,y), axis=1))

print(data)

x_train = data[:70,0]
y_train = data[:70,1]
x_test = data[70:,0]
y_test = data[70:,1]