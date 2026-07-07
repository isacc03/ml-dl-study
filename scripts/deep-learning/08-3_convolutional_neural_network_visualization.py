# 08-3 합성곱 신경망의 시각화
# Source: https://colab.research.google.com/drive/1gdcunVMIEHBhjHrBGVWomjK9Ae27MShu?usp=sharing
# Generated from the sanitized notebook.

# # 합성곱 신경망의 시각화

# %% Cell 1
# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 텐서플로 연산을 결정적으로 만듭니다.
import keras
import tensorflow as tf

keras.utils.set_random_seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

# ## 가중치 시각화

# %% Cell 2
# 코랩에서 실행하는 경우에는 다음 명령을 실행하여 best-cnn-model.keras 파일을 다운로드받아 사용하세요.
from urllib.request import urlretrieve
urlretrieve('https://github.com/rickiepark/hg-mldl2/raw/main/best-cnn-model.keras', 'best-cnn-model.keras')

# %% Cell 3
import keras
model = keras.models.load_model('best-cnn-model.keras')

# %% Cell 4
print(model.layers)

# %% Cell 5
conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape)

# %% Cell 6
conv_weights = conv.weights[0].numpy()

print(conv_weights.mean(), conv_weights.std())

# %% Cell 7
import matplotlib.pyplot as plt

plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# %% Cell 8
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# %% Cell 9
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Input(shape=(28,28,1)))
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                                          padding='same'))

# %% Cell 10
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)

# %% Cell 11
no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())

# %% Cell 12
plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# %% Cell 13
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# ## 함수형 API

# %% Cell 14
inputs = keras.Input(shape=(784,))
dense1 = keras.layers.Dense(100, activation='relu')
dense2 = keras.layers.Dense(10, activation='softmax')

hidden = dense1(inputs)
outputs = dense2(hidden)

func_model = keras.Model(inputs, outputs)

# %% Cell 15
print(model.inputs)

# %% Cell 16
conv_acti = keras.Model(model.inputs[0], model.layers[0].output)

# ## 특성 맵 시각화

# %% Cell 17
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()

# %% Cell 18
plt.imshow(train_input[0], cmap='gray_r')
plt.show()

# %% Cell 19
ankle_boot = train_input[0:1].reshape(-1, 28, 28, 1)/255.0
feature_maps = conv_acti.predict(ankle_boot)

# %% Cell 20
print(feature_maps.shape)

# %% Cell 21
fig, axs = plt.subplots(4, 8, figsize=(15,8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')
plt.show()

# %% Cell 22
conv2_acti = keras.Model(model.inputs, model.layers[2].output)

# %% Cell 23
feature_maps = conv2_acti.predict(ankle_boot)

# %% Cell 24
print(feature_maps.shape)

# %% Cell 25
fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')
plt.show()

# # 합성곱 신경망의 시각화 (파이토치)

# %% Cell 26
# 실행마다 동일한 결과를 얻기 위해 파이토치에 랜덤 시드를 지정하고 GPU 연산을 결정적으로 만듭니다.
import torch

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

# %% Cell 27
# 코랩에서 실행하는 경우에는 다음 명령을 실행하여 best_cnn_model.pt 파일을 다운로드받아 사용하세요.
from urllib.request import urlretrieve
urlretrieve('https://github.com/rickiepark/hg-mldl2/raw/refs/heads/main/best_cnn_model.pt', 'best_cnn_model.pt')

# %% Cell 28
import torch.nn as nn

model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding='same'))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(2))
model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding='same'))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(2))
model.add_module('flatten', nn.Flatten())
model.add_module('dense1', nn.Linear(3136, 100))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(0.3))
model.add_module('dense2', nn.Linear(100, 10))

# %% Cell 29
model.load_state_dict(torch.load('best_cnn_model.pt', weights_only=True))

# %% Cell 30
layers = [layer for layer in model.children()]

# %% Cell 31
print(layers[0])

# %% Cell 32
print(model[0])

# %% Cell 33
for name, layer in model.named_children():
    print(f"{name:10s}", layer)

# %% Cell 34
print(model.conv1)

# %% Cell 35
conv_weights = model.conv1.weight.data
print(conv_weights.mean(), conv_weights.std())

# %% Cell 36
import matplotlib.pyplot as plt

plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# %% Cell 37
print(conv_weights.shape)

# %% Cell 38
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[i*16 + j,0,:,:], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# %% Cell 39
from torchvision.datasets import FashionMNIST

fm_train = FashionMNIST(root='.', train=True, download=True)
train_input = fm_train.data

# %% Cell 40
plt.imshow(train_input[0], cmap='gray_r')
plt.show()

# %% Cell 41
ankle_boot = train_input[0:1].reshape(1, 1, 28, 28) / 255.0

model.eval()
with torch.no_grad():
    feature_maps = model.conv1(ankle_boot)
    feature_maps = model.relu1(feature_maps)

# %% Cell 42
print(feature_maps.shape)

# %% Cell 43
fig, axs = plt.subplots(4, 8, figsize=(15,8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,i*8 + j,:,:])
        axs[i, j].axis('off')
plt.show()

# %% Cell 44
model.eval()
with torch.no_grad():
    feature_maps = model.conv1(ankle_boot)
    feature_maps = model.relu1(feature_maps)
    feature_maps = model.pool1(feature_maps)
    feature_maps = model.conv2(feature_maps)
    feature_maps = model.relu2(feature_maps)

# %% Cell 45
model.eval()
x = ankle_boot
with torch.no_grad():
    for name, layer in model.named_children():
        x = layer(x)
        if name == 'relu2':
            break
feature_maps = x

# %% Cell 46
fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,i*8 + j,:,:])
        axs[i, j].axis('off')
plt.show()
