# 08-2 합성곱 신경망을 사용한 이미지 분류
# Source: https://colab.research.google.com/drive/1qb1_cyo-QTk8QDj6NjMsj3qU8rz79tCV?usp=sharing
# Generated from the sanitized notebook.

# %% Cell 1
# 패션 mnist 데이터 불러오기
import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

# %% Cell 2
# 합성곱 신경망 모델 구조 만들기
model = keras.Sequential()
model.add(keras.layers.Input(shape=(28,28,1)))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))

# %% Cell 3
# (2,2)풀링
model.add(keras.layers.MaxPooling2D(2))

# %% Cell 4
# 두 번째 합성곱-풀링층 추가하기
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

# %% Cell 5
# 평탄화층이랑 출력층 추가하기
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

# %% Cell 6
# 모델 구조 확인하기
model.summary()

# %% Cell 7
# 시각적으로 표현해보기
keras.utils.plot_model(model)

# %% Cell 8
# 입력, 출력도 표시해서 시각화
keras.utils.plot_model(model, show_shapes=True)

# %% Cell 9
# 모델 컴파일하고 훈련하기
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

# %% Cell 10
# epoch당 loss 출력하기
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# %% Cell 11
# 모델 평가하기
model.evaluate(val_scaled, val_target)

# %% Cell 12
# 새로운 데이터에 대한 예측 : 첫 번째 샘플을 처음 본 이미지라고 가정
plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

# %% Cell 13
preds = model.predict(val_scaled[0:1])
print(preds)

# %% Cell 14
plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

# %% Cell 15
# 레이블을 출력하기 위한 클래스 리스트 만들기
classes = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']

# %% Cell 16
# 예측 결과 출력하기
import numpy as np
print(classes[np.argmax(preds)])

# %% Cell 17
# 모델의 일반화 성능을 확인하기 위해 테스트 세트 스케일링하기
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

# %% Cell 18
# 테스트 세트로 모델 평가하기
model.evaluate(test_scaled, test_target)

# %% Cell 19
# 전체 코드 리뷰
## 패션 MNIST 데이터 불러오기
import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

### 합성곱 신경망의 입력 형식에 맞게 데이터의 모양을 변경, 픽셀 값을 0~1사이로 정규화
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

## 합성곱 신경망 만들기
model = keras.Sequential()

model.add(keras.layers.Input(shape=(28,28,1)))

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))

model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))

model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

### 모델 구조 확인

model.summary()
keras.utils.plot_model(model)
keras.utils.plot_model(model, show_shapes=True, to_file='cnn_architecture.png', dpi=300)

## 모델 컴파일과 훈련
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

### 콜백 모델, 조기종료 저장하기
checkpoint_cb = keras.callbacks.ModelCheckpoint('best_cnn_model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

model.evaluate(val_scaled, val_target)

## 모델의 일반화 성능 확인하기
### 첫 번째 샘플 데이터 활용하여 모델 예측 확인해보기
plt.imshow(val_scaled[0].reshape(28,28), cmap='gray_r')
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)

plt.bar(range(1,11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

classes = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌달', '셔츠', '스니커즈', '가방', '부츠']

### 테스트 세트 이용해서 모델 성능 평가하기
import numpy as np
print(classes[np.argmax(preds)])

test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

model.evaluate(test_scaled, test_target)
