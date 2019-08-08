
## 导入数据集
载入并准备好MNIST数据集。将样本从整数转换为浮点数：
```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```
## 模型堆叠
将模型的各层堆叠起来，以搭建 tf.keras.Sequential 模型。为训练选择优化器和损失函数：
```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
```
 tf.keras.layers.Flatten(input_shape=(28, 28))
 
 将向量打平，28*28 -> 784
## Dropout
keras.layers.Dropout(rate, noise_shape=None, seed=None)
将 Dropout 应用于输入。

Dropout 包括在训练中每次更新时， 将输入单元的按比率随机设置为 0， 这有助于防止过拟合。

### 参数

rate: 在 0 和 1 之间浮动。需要丢弃的输入比例。
noise_shape: 1D 整数张量， 表示将与输入相乘的二进制 dropout 掩层的形状。 例如，如果你的输入尺寸为 (batch_size, timesteps, features)，然后 你希望 dropout 掩层在所有时间步都是一样的， 你可以使用 noise_shape=(batch_size, 1, features)。
seed: 一个作为随机种子的 Python 整数。
'''
## 训练并验证模型：
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
```
