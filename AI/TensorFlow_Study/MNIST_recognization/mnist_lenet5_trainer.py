import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape, name=None):
    """
    Initialize the weight of previous shape by random
    :param shape: shape of weight
    :param name: name of weight
    :return: initial weight
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape):
    """
    Initialize the bias of previous shape by random
    :param shape: shape of bias
    :return: initial bias
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    convoltional computation
    :param x: target
    :param W: convoltion kernel
    :return: result of convoltional computation
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    pooling computation
    :param x: target to pool
    :return:result of pooling computation
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# get data set
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# input
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')

# output
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y')

x_image = tf.reshape(x, [-1, 28, 28, 1])

# Layer1  convoltional layer with 32 features
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# Layer2  pooling layer
h_pool1 = max_pool_2x2(h_conv1)

# Layer3  convoltional layer with 64 features
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Layer4  pooling layer
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# Layer5 full connection layer with 1024 features
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Layer6 dropout layer
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Layer7 full connection layer with 10 features
W_fc2 = weight_variable([1024, 10], "weight")
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# cross entropy loss
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# train step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# result
rst = tf.argmax(y_conv, 1, name="rst")

# correct prediction
correct_prediction = tf.equal(rst, tf.argmax(y_, 1))

# accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        x_batch, y_batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy  = accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
            test_accuracy =  accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            print('step %d, training accuracy %g, test accuracy %g' % (i, train_accuracy , test_accuracy))
        train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})
    saver.save(sess, './model/data.chkp')
