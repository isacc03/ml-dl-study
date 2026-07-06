# 08-3 합성곱 신경망의 시각화
# Source: https://colab.research.google.com/drive/1gdcunVMIEHBhjHrBGVWomjK9Ae27MShu?usp=sharing
# Generated from the sanitized notebook.

# %% Cell 1
import keras

model = keras.models.load_model('best-cnn-model.keras')

# %% Cell 2
print(model.layers)

# %% Cell 3
conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape)

# %% Cell 4
conv_weights = conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std())

# %% Cell 5
import matplotlib.pyplot as plt

plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# %% Cell 6
fig, axs = plt.subplots(2, 16, figsize=(15, 2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:, :, 0, i * 16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# %% Cell 7
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Input(shape=(28, 28, 1)))
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))

# %% Cell 8
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)

# %% Cell 9
no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())

# %% Cell 10
plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# %% Cell 11
fig, axs = plt.subplots(2, 16, figsize=(15, 2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:, :, 0, i * 16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# %% Cell 12
inputs = keras.Input(shape=(784,))
dense1 = keras.layers.Dense(100, activation='relu')
dense2 = keras.layers.Dense(10, activation='softmax')

# %% Cell 13
hidden = dense1(inputs)

# %% Cell 14
outputs = dense2(hidden)

# %% Cell 15
func_model = keras.Model(inputs, outputs)

# %% Cell 16
print(model.inputs)

# %% Cell 17
conv_acti = keras.Model(model.inputs[0], model.layers[0].output)

# %% Cell 18
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
plt.imshow(train_input[0], cmap='gray_r')
plt.show()

# %% Cell 19
ankle_boot = train_input[0:1].reshape(-1, 28, 28, 1) / 255.0
feature_maps = conv_acti.predict(ankle_boot)

# %% Cell 20
print(feature_maps.shape)

# %% Cell 21
fig, axs = plt.subplots(4, 8, figsize=(15, 8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0, :, :, i * 8 + j])
        axs[i, j].axis('off')
plt.show()

# %% Cell 22
conv2_acti = keras.Model(model.inputs[0], model.layers[2].output)

# %% Cell 23
feature_maps = conv2_acti.predict(ankle_boot)

# %% Cell 24
print(feature_maps.shape)

# %% Cell 25
fig, axs = plt.subplots(8, 8, figsize=(12, 12))
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0, :, :, 8 * i + j])
        axs[i, j].axis('off')
plt.show()

# %% Cell 26
from urllib.request import urlretrieve

urlretrieve('https://bit.ly/3DQeEH8', 'best_cnn_model.pt')

# %% Cell 27
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

# %% Cell 28
import torch

model.load_state_dict(torch.load('best_cnn_model.pt', weights_only=True))

# %% Cell 29
layers = [layer for layer in model.children()]

# %% Cell 30
print(layers[0])

# %% Cell 31
print(model[0])

# %% Cell 32
for name, layer in model.named_children():
    print(f"{name:10s}", layer)

# %% Cell 33
print(model.conv1)

# %% Cell 34
conv_weights = model.conv1.weight.data
print(conv_weights.mean(), conv_weights.std())

# %% Cell 35
import matplotlib.pyplot as plt

plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# %% Cell 36
print(conv_weights.shape)

# %% Cell 37
fig, axs = plt.subplots(2, 16, figsize=(15, 2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[16 * i + j, 0, :, :], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# %% Cell 38
from torchvision.datasets import FashionMNIST

fm_train = FashionMNIST(root='.', train=True, download=True)
train_input = fm_train.data
plt.imshow(train_input[0], cmap='gray_r')
plt.show()

# %% Cell 39
ankle_boot = train_input[0:1].reshape(-1, 1, 28, 28) / 255.0

model.eval()
with torch.no_grad():
    feature_maps = model.conv1(ankle_boot)
    feature_maps = model.relu1(feature_maps)

# %% Cell 40
print(feature_maps.shape)

# %% Cell 41
fig, axs = plt.subplots(4, 8, figsize=(15, 8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0, i * 4 + j, :, :])
        axs[i, j].axis('off')
plt.show()

# %% Cell 42
model.eval()
with torch.no_grad():
    feature_maps = model.conv1(ankle_boot)
    feature_maps = model.relu1(feature_maps)
    feature_maps = model.pool1(feature_maps)
    feature_maps = model.conv2(feature_maps)
    feature_maps = model.relu2(feature_maps)

# %% Cell 43
model.eval()
x = ankle_boot
with torch.no_grad():
    for name, layer in model.named_children():
        x = layer(x)
        if name == 'relu2':
            break
feature_maps = x

# %% Cell 44
fig, axs = plt.subplots(8, 8, figsize=(12, 12))
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0, i * 8 + j, :, :])
        axs[i, j].axis('off')
plt.show()
