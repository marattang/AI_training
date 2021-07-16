import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) 흑백데이터이기 때문에 3차원
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print("x[0] 값: ", x_train[0])
print("y[0] 값: ", y_train[0])

plt.imshow(x_train[0], 'gray') # 깨진 컬러로 나오기 때문에 색 지정


plt.show()