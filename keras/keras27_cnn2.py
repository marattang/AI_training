from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
# 이미지는 Conv2D, CNN이라고 생각하면 된다.

model = Sequential()                                             # (N(batch_size, row), 5, 5, 1)
model.add(Conv2D(10, kernel_size=(2, 2),                  # (N, 10, 10, 1)
                padding='same', input_shape=(10, 10, 1))) # (N, 10, 10, 10) 
model.add(Conv2D(20, (2, 2), activation='relu'))          # (N, 9, 9, 20)
model.add(Conv2D(30, (2,2), padding='valid'))             # (N, 8, 8, 30)
model.add(MaxPooling2D())                                 # (N, 4, 4, 30)
model.add(Conv2D(15, (2,2)))                              # (N, 3, 3, 15)
model.add(Flatten())                                      # (N, 135)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 노드의 갯수를 늘리면 그만큼 연산이 늘어난다.

model.summary()

# kernel_size = 자르는 크기

# 과제 1. Conv2D의 디폴트 엑티베이션
# 과제 2. Conv2D summary의 파라미터 갯수 완벽 이해