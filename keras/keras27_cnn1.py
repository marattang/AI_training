from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras import activations
# 이미지는 Conv2D, CNN이라고 생각하면 된다.

model = Sequential()                                             # (N(batch_size, row), 5, 5, 1)
model.add(Conv2D(10, kernel_size=(2, 2), padding='same', input_shape=(5, 5, 1))) # (N, 4, 4, 10)
model.add(Conv2D(20, (2, 2), activation='relu')) # kernel size 생략 2,2 로 자른다. # (N, 3, 3, 20)
model.add(Conv2D(30, (2,2), padding='valid')) # padding의 기본값은 valid다. same은 채워준다는 뜻이다.
model.add(Flatten())          # (N, 180)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 노드의 갯수를 늘리면 그만큼 연산이 늘어난다.

model.summary()

# kernel_size = 자르는 크기

# 과제 1. Conv2D의 디폴트 엑티베이션
# 과제 2. Conv2D summary의 파라미터 갯수 완벽 이해