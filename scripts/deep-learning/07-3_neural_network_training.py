# 07-3 신경망 모델 훈련
# Source: https://colab.research.google.com/drive/1_NKY6AH8dSi-gtJ0VVImct5X7VdRhrMa?usp=sharing
# Generated from the sanitized notebook.

# %% Cell 1
import keras
import tensorflow as tf

# 교재와 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 텐서플로 연산을 결정적으로 만든다.
keras.utils.set_random_seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

# %% Cell 2
from sklearn.model_selection import train_test_split
# fashion_mnist 데이터 불러오기
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 스케일링 하기
train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

# %% Cell 3
# 모델 생성 함수
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28,28)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

# %% Cell 4
# 생성 후 구조 확인하기
model = model_fn()
model.summary()

# %% Cell 5
# 모델 구조 만들기
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 모델 훈련하고 손실값을 표시하기 위해 객체 만들기
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)

# %% Cell 6
# 모델 객체에 들어있는 key값 확인하기
print(history.history.keys())

# %% Cell 7
# epoch에 따른 손실값 출력하기
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %% Cell 8
# epoch에 따른 정확도값 출력하기
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# %% Cell 9
# epoch를 20으로 늘려서 훈련하기
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=20, verbose=0)

# %% Cell 10
# 손실값 출력하기
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %% Cell 11
# 검증 손실값까지 계산할 수 있게 모델 훈련하기
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))

# %% Cell 12
# 검증 손실까지 계산돼있는지 확인하기
print(history.history.keys())

# %% Cell 13
# 손실값 모두 출력하기
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# %% Cell 14
# optimizer도 설정해서 모델 훈련하기
model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))

# %% Cell 15
# 다시 출력하기
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# %% Cell 16
# 드롭아웃층 만들기
model = model_fn(keras.layers.Dropout(0.3))
model.summary()

# %% Cell 17
# 드롭아웃층 만든 상태로 다시 모델 훈련하기
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))

# %% Cell 18
# 손실값 출력하기
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# %% Cell 19
# 모델 구조와 파라미터를 모두 저장
model.save('model_whole.keras')

# %% Cell 20
# 훈련된 모델의 파라미터만 저장
model.save_weights('model.weights.h5')

# %% Cell 21
# 두 파일이 잘 저장되었는지 확인하기
# In Colab/Jupyter: !ls -al model*

# %% Cell 22
# 새로운 모델을 만들고 이전에 저장한 모델 파라미터 적재
model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model.weights.h5')

# %% Cell 23
# 모델의 정확도를 직접 계산하기
import numpy as np

val_labels = np.argmax(model.predict(val_scaled), axis=-1)
print(np.mean(val_labels == val_target))

# %% Cell 24
# 저장한 모델 자체를 불러와서 정확도랑 검증 손실 출력하기
model = keras.models.load_model('model_whole.keras')
model.evaluate(val_scaled, val_target)

# %% Cell 25
# 콜백을 이용하여 최상의 검증 점수를 낸 모델을 저장
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb])

# %% Cell 26
# 저장한 모델을 불러와서 정확도와 검증 손실 출력
model = keras.models.load_model('best_model.keras')
model.evaluate(val_scaled, val_target)

# %% Cell 27
# 조기종료 콜백을 이용하여 필요한 만큼만 모델 훈련하기
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

# %% Cell 28
# 마지막으로 훈련한 epoch 출력하기
print(early_stopping_cb.stopped_epoch)

# %% Cell 29
# epoch당 일반 손실과 검증 손실을 시각화하기
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# %% Cell 30
# 모델 검증 점수 확인하기
model.evaluate(val_scaled, val_target)
