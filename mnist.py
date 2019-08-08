#conde:utf-8
'''
手写数字识别
'''
import  tensorflow as tf
import  os
from tensorflow import  keras
from tensorflow.keras import datasets,layers,optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
(xs,ys),_ =  datasets.mnist.load_data()
xs = tf.convert_to_tensor(xs,dtype=tf.float32) / 255
ys = tf.convert_to_tensor(ys,dtype=tf.int32)
ys = tf.one_hot(ys,depth=10)
print('datasets',xs.shape,ys.shape)
db = tf.data.Dataset.from_tensor_slices((xs,ys))
db = db.batch(200)

"""
for step,(x,y) in enumerate(db):
    print(step,x.shape,y,y.shape)
"""
model = keras.Sequential([
    layers.Dense(512,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(10)])
optimizer = optimizers.SGD(learning_rate=0.001)

def train_epoch(epoch):
    for step,(x,y) in enumerate(db):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28 * 28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch,step,'loss:',loss.numpy())
def train():
    for epoch in range(30):
        train_epoch(epoch)
if __name__ == '__main__':
    train()



