#conding:utf-8
# 导入TensorFlow和tf.keras
import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras import datasets

from keras.datasets import boston_housing
# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt

#(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

(train_images, train_labels), (test_images, test_labels) = boston_housing.load_data()
class_names = ['0','1','2','3','4','5','6','7','8','9']

#train_images = train_images / 255.0
#test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
