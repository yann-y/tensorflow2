#coding:utf-8
#数据分析库
import pandas as pd
import  tensorflow as tf
from tensorflow  import  keras
import  os
#科学计算库
import numpy as np
from pandas import Series,DataFrame

train_x = np.loadtxt("train.csv",delimiter=",",dtype=np.float)
train_y = train_x[:,0:1]
print(train_x.shape)
print(train_y.shape)
x = train_x[:,2:]
y = train_y
print(x.shape)

# 转换为Tensor
train_x = tf.convert_to_tensor(x,dtype=tf.float32)
train_y = tf.convert_to_tensor(y,dtype=tf.int32)

print(x.shape,y.shape,x.dtype,y.dtype)
print(tf.reduce_min(x),tf.reduce_max(x))
print(tf.reduce_min(y),tf.reduce_max(y))


# [b,728] => [b,512] => [b,128]  => [b,10]
w1 = tf.Variable(tf.random.truncated_normal([7,16],stddev=0.1))
b1 = tf.Variable(tf.zeros([16]))
w2= tf.Variable(tf.random.truncated_normal([16,4],stddev=0.1))
b2 = tf.Variable(tf.zeros([4]))
w3 = tf.Variable(tf.random.truncated_normal([4,1],stddev=0.1))
b3 = tf.Variable(tf.zeros([2]))
lr = 1e-3
# h1 = x@w1 + b1
for epoch in range(1):
    for i in range(0,890):
        # x:[128,28,28]
        # y: [128]
        # x: [b,28*28]
        # h1 = x@w1 + b1;
        # [b.784]@[784,256] + [256] =>  [b,256] +[256] => b[256] + [256]
        # x = train_x[i:i+2,:]
        x = tf.broadcast_to(train_x[i,:],[1,7])
        x = tf.Variable(x)
        #print(x)
        #(w1)
        y = train_y[i]
        with tf.GradientTape() as  tape:
            h1 = x@w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2
            out = h2@w3 + b3

            # compute loss
            # out[b,10]
            # y: [b] => [b,10]
            y_onehot = tf.one_hot(y,depth=2)
            # mes = mean((y-out)^2)
            # [b,10]
            loss = tf.square(y_onehot - out)
            # mean: scalar
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        '''
        w1 = w1 - lr * grads[0]
        b1 =b1 - lr * grads[1]
        w2 = w2 - lr * grads[2]
        b2 = b2 - lr * grads[3]
        w3 = w3 - lr * grads[4]
        b3 = b3 - lr * grads[5]
        '''
        if i % 100 == 0:
            print(epoch,i,'loss = ',float(loss))

    print(epoch,'loss = ',float(loss))

test = np.loadtxt("test.csv",delimiter=",",dtype=np.float)
text_x = test[:,1:]
text_x = tf.convert_to_tensor(text_x,dtype=tf.float32)
text_y=tf.Variable(tf.zeros([491]))
for i in range(0,419):
    x = tf.broadcast_to(text_x[i,:],[1,7])
    x = tf.Variable(x)
    #print(x)
    #print(w1)
    with tf.GradientTape() as  tape:
        h1 = x@w1 + b1
        h1 = tf.nn.relu(h1)
        h2 = h1@w2 + b2
        h2 = tf.nn.relu(h2)
        out = h2@w3 + b3
        out = tf.nn.relu(out)
        print(out)
