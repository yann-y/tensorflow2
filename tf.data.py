import  tensorflow as tf
from tensorflow  import  keras
from  tensorflow.keras import datasets
import  os
import pandas as pd
import  tensorflow as tf
from tensorflow  import  keras
import  os
#科学计算库
import numpy as np
import cv2
from PIL import Image
from pandas import Series,DataFrame

csv_path = "E:\study\yolov3\DogAndCat0_FlyAI\data\input\dev.csv"
data_train = pd.read_csv(csv_path)
print(data_train.loc[0,'image_path'])
#print(data_train)
#mg = [100,500,380,3]
img =np.zeros(shape=(100,280,280))
print(data_train.shape[0])
train_x = []
y =np.zeros(100)
for i in range(data_train.shape[0]):
    imageFile = "E:\study\yolov3\DogAndCat0_FlyAI\data\input\\" + data_train.loc[i, 'image_path']
    reimg = cv2.imread(imageFile,0)
    #print(reimg.shape)
    # 把图片的大小统一更改为280；
    resized = cv2.resize(reimg,(280,280), interpolation=cv2.INTER_NEAREST)
    #print(resized.shape)
    img[i] = resized
    #print(data_train.loc[i, 'label'])
    #y[i] = data_train.loc[i, 'label']
    y[i] = data_train.loc[i, 'label']
print(img.shape,y.shape)
def img_convert(imagePath,data):
    imageFile = imagePath + data.loc[i, 'image_path']
    reimg = cv2.imread(imageFile, 0)
    # print(reimg.shape)
    # 把图片的大小统一更改为280；
    resized = cv2.resize(reimg, (280, 280), interpolation=cv2.INTER_NEAREST)
    # print(resized.shape)
    img[i] = resized
    # print(data_train.loc[i, 'label'])
    # y[i] = data_train.loc[i, 'label']
    y[i] = data.loc[i, 'label']
    return img,y;
train_images = tf.convert_to_tensor(img)
train_labels = tf.convert_to_tensor(y)
#test_img , test_labels = img_convert()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(280, 280)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)