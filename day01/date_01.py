#conding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import pylab
import tensorflow as tf
from tensorflow import keras

#查看TensorFlow版本
print("TensorFlow的版本为："+tf.__version__)
"""
img=plt.imread('../te.jpg')
print(img.shape)
plt.figure()
plt.imshow(img)
plt.colorbar()
pylab.show()
plt.imsave('../te.png',img)
"""

#加载训练集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10,10))#图片大小10*10
train_images = train_images / 255.0
test_images = test_images / 255.0
for i in range(25):
    plt.subplot(5,5,i+1)#有一个五行五列的画图区域，i+1现在画第i+1个图
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])









