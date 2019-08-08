import  tensorflow as tf
from tensorflow  import  keras
from  tensorflow.keras import datasets
import  os
import pandas as pd
import  tensorflow as tf
from tensorflow  import  keras
import  os
import  random
#科学计算库
import numpy as np
import cv2
from PIL import Image
from pandas import Series,DataFrame
def img_convert(imagePath,lable):

    if lable == "cat":
        for i in range(2000):
            imageFile = imagePath + lable + "." + str(i) + ".jpg"
            img[i] = cv2.imread(imageFile,0)
    else:
        j = 2000
        for i in range(2000):
            imageFile = imagePath + lable + "." + str(i) + ".jpg"
            img[j] = cv2.imread(imageFile,0)
            y[j] = 1
            j += 1
def img_test(imagePath,lable):

    if lable == "cat":
        j = 0
        for i in range(12000,12500):
            imageFile = imagePath + lable + "." + str(i) + ".jpg"
            test_x[j] = cv2.imread(imageFile,0)
    else:
        j = 500
        for i in range(12000,12500):
            imageFile = imagePath + lable + "." + str(i) + ".jpg"
            test_x[j] = cv2.imread(imageFile,0)
            test_y[j] = 1
            j += 1
img =np.zeros(shape=(4000,224,224),dtype=float)
y = np.zeros(shape=(4000))
test_x = np.zeros(shape=(1000,224,224),dtype=float)
test_y = np.zeros(shape=(1000))
img_convert("D:\BaiduNetdiskDownload\kaggle\\train2\\", "cat")
img_convert("D:\BaiduNetdiskDownload\kaggle\\train2\\", "dog")
img_test("D:\BaiduNetdiskDownload\kaggle\\train2\\", "cat")
img_test("D:\BaiduNetdiskDownload\kaggle\\train2\\", "dog")
print("data input .........  /input/train2/")
img = tf.convert_to_tensor(img,dtype=tf.float32) /225.0
y = tf.convert_to_tensor(y,dtype=tf.float32)
test_x = tf.convert_to_tensor(test_x,dtype=tf.float32) /255.0
test_y = tf.convert_to_tensor(test_y,dtype=tf.float32)
def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    #dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(4000).batch(batch_size) #每次随机调整数据顺序
    return dataset
datasets_train = train_input_fn(img,y,2048)
datasets_test = tf.data.Dataset.from_tensor_slices((test_x,test_y))
datasets_test = datasets_test.shuffle(1000)



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(224, 224)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(datasets_train, epochs=5)
test_loss, test_acc = model.evaluate(datasets_test)

print('\nTest accuracy:', test_acc)

