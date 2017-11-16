#!/usr/bin/env
#****UTF-8****
"""
TensorFlow
卷积神经网络 CNN
MNIST手写图像识别

Author        :TensorFlow
Comment Adder :王振荟
Link:http://www.soaringroad.com/?p=115
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


"""
根据给定的维度参数，初始化weight，正态分布随机
"""
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

"""
根据给定的维度参数，初始化bias，初始值0.1
"""
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

"""
卷积计算定义
"""
def conv2d(x, W):
  """
  strides=[1,1,1,1]定义了卷积步长为1，
  padding='SAME'定义了padding，在图像四周添加0，使得卷积后，大小和原来相同28*28
  """
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

"""
2*2池化计算定义
"""
def max_pool_2x2(x):
  """
  ksize=[1, 2, 2, 1]定义了池化大小为2*2，
  strides=[1, 2, 2, 1]定义了x方向步长为2，y方向步长为2，池化大小为2*2，所以，意思就是不重复的方式进行池化，
  padding='SAME'定义了padding，与上面conv2d的padding不同，请参考官方API，说得比较清楚
  """
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #获取MNIST数据集
x = tf.placeholder(tf.float32, shape=[None, 784],name='x') #定义输入参数x  784=28*28
y_ = tf.placeholder(tf.float32, shape=[None, 10],name='y_') #定义输入参数y_
x_image = tf.reshape(x, [-1, 28, 28, 1]) #将x转成28*28的格式

# Layer1  第一卷积层 要获得32个特征图
W_conv1 = weight_variable([5, 5, 1, 32]) #定义第一个卷积层的w,卷积核大小是5*5,输入数据的特征图数量为1，该层输出数据的特征图数量为32，所以是5,5,1,32
b_conv1 = bias_variable([32]) #定义第一个卷积层的b,因为一个特征图内偏置共享（参考CNN的理论说明），所以b的个数与输出数据的特征图数量相同32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#卷积计算，之后的结果，使用relu作为激活函数再计算  输出结果应该是28*28*32(特征图数)

# Layer2  第一池化层
h_pool1 = max_pool_2x2(h_conv1)#池化计算 输出结果应该是14*14*32，参考上面max_pool_2x2方法的说明

# Layer3  第二卷积层 要获得64个特征图
W_conv2 = weight_variable([5, 5, 32, 64]) #定义第二个卷积层的w,卷积核大小是5*5,输入数据（上一层的输出数据）的特征图数量为1，该层输出数据的特征图数量为32，所以是5,5,32,64
b_conv2 = bias_variable([64])#定义第二个卷积层的b,因为一个特征图内偏置共享（参考CNN的理论说明），所以b的个数与输出数据的特征图数量相同64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #卷积计算，之后的结果，使用relu作为激活函数再计算  输出结果应该是28*28*64(特征图数)

# Layer4  第二池化层
h_pool2 = max_pool_2x2(h_conv2) #池化计算 输出结果应该是7*7*64，参考上面max_pool_2x2方法的说明
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#把结果转成一个数组形式的，相当于有3133=7*7*64个特征值

# Layer5 第一全连接层 要获得1024个特征《值》，是值...是值...(重要的事情说三遍)
W_fc1 = weight_variable([7 * 7 * 64, 1024]) #前一层的特征数3133=7*7*64,想要获得1024个特征，因为是全连结，所以是[7 * 7 * 64, 1024]
b_fc1 = bias_variable([1024])#和普通的神经网络一样，想要获得1024个特征值，所以每一个都要有一个偏置，所以是1024
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)#神经网络的正向传播，结果使用relu作为激活函数再计算(不懂的请参照最基本的神经网络学习吧)

# Layer6 Dropout层
keep_prob = tf.placeholder(tf.float32,name='keep_prob')#定义结果保留率
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#将一部分结果舍弃，即dropout

# Layer7 第二全连接层 要获得10个特征《值》，至于为什么是10，因为0123456789一共是个数字.....
W_fc2 = weight_variable([1024, 10]) #参考第一全连接层
b_fc2 = bias_variable([10]) #参考第一全连接层
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 #参考第一全连接层 正向传播


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))#定义交叉熵的计算方式 这里是softmax交叉熵


train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#定义梯度，学习率0.0004，最小化熵的梯度求解

rst=tf.argmax(y_conv,1,name="rst")


correct_prediction = tf.equal(rst, tf.argmax(y_, 1))#定义正确预测的判定


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#计算准确率

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())#全局初始化变量
  for i in range(2000): #定义训练次数2000
    batch = mnist.train.next_batch(50) #获取下50条数据作为batch训练数据，变相定义batch大小50
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy)) #每一百步，输出准确率
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})#使用batch训练数据进行训练，保留率50%
  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})) #输出测试数据，准确率

  saver.save(sess,'./data.chkp')
