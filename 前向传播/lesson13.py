#conding:utf-8
import  tensorflow as tf
from tensorflow  import  keras
from  tensorflow.keras import datasets
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
'''
0 全部输出
2 仅输出错误信息
'''
'''
loss = nan 梯度爆炸'''
# x:[60k,28,28]
# y: [60k]
(x,y),(x_test,y_test) = datasets.mnist.load_data()
# 转换为Tensor
x = tf.convert_to_tensor(x,dtype=tf.float32) / 255
y = tf.convert_to_tensor(y,dtype=tf.int32)

x_test = tf.convert_to_tensor(x_test,dtype=tf.float32) / 255
y_test = tf.convert_to_tensor(y_test,dtype=tf.int32)
print(x.shape,y.shape,x.dtype,y.dtype)
print(tf.reduce_min(x),tf.reduce_max(x))
print(tf.reduce_min(y),tf.reduce_max(y))

train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(128)
train_iter = iter(train_db) # 生成迭代器
sample = next(train_iter)
print('batch :',sample[0].shape)

# [b,728] => [b,512] => [b,128]  => [b,10]
w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2= tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
lr = 1e-3
# h1 = x@w1 + b1
for epoch in range(10):
    for step,(x,y) in enumerate(train_db):
        # x:[128,28,28]
        # y: [128]
        # x: [b,28*28]
        # h1 = x@w1 + b1;
        # [b.784]@[784,256] + [256] =>  [b,256] +[256] => b[256] + [256]
        x = tf.reshape(x,[-1,28*28])
        with tf.GradientTape() as  tape:
            h1 = x@w1 + tf.broadcast_to(b1,[x.shape[0],256])
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2
            out = h2@w3 + b3

            # compute loss
            # out[b,10]
            # y: [b] => [b,10]
            y_onehot = tf.one_hot(y,depth=10)
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
        if step % 100 == 0:
            print(epoch,step,'loss = ',float(loss))

    total_correct = 0
    total_num = 0
    for  step,(x,y) in enumerate(test_db):
        x = tf.reshape(x,[-1,28*28])
        h1 = tf.nn.relu(x@w1+b1)
        h2 = tf.nn.relu(h1@w2 + b2)
        out = h2@w3 + b3
        # int 64

        prob = tf.nn.softmax(out,axis=1)
        pred = tf.argmax(prob,axis=1)
        #print(pred.shape,out.shape,prob.shape)
        pred = tf.cast(pred,dtype=tf.int32)
        #print(pred.shape,y.shape)
        correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_correct += int(correct)
        total_num += x.shape[0]
    acc = total_correct / total_num
    print("test acc: ",acc)