from tensorflow.keras.datasets import cifar100

# 완성하시오 imshow 이미지 확인 mnist1 카피해서 확인

import numpy as np
import matplotlib.pyplot as plt

# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1) 컬러데이터이기 때문에 3차원
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1) y 6개

print("x[0] 값: ", x_train[0])
print("y[0] 값: ", y_train[0])

plt.imshow(x_train[0], 'gray') # 깨진 컬러로 나오기 때문에 색 지정


plt.show()
print("#!@$@!%!@#", np.unique(y_train))