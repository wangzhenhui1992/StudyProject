# -*- utf-8 -*-

import tensorflow as tf
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
import reader


def weight_variable(shape):
    print("exec weight_variable")
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial)


def load_img(img_path):
    print("exec load_img")
    img = image.load_img(img_path, target_size=(224, 224))  # 224×224
    x = image.img_to_array(img)  # 三维(3, 224, 224)
    x = np.expand_dims(x, axis=0)  # 四维(1, 3, 224, 224)
    x = preprocess_input(x)
    return x


def predict_layer(model, x):
    print("exec predict_layer")
    y_pred = model(x)
    return y_pred


def content_loss_layer(A, B):
    print("exec content_loss_layer")
    return tf.reduce_sum(tf.square(A - B))


def predict(x):
    print("exec predict")
    block5_conv1 = predict_layer(block5_conv1_model, x)
    block5_conv2 = predict_layer(block5_conv2_model, x)
    block5_conv3 = predict_layer(block5_conv3_model, x)

    block5_conv1 = tf.transpose(tf.reshape(block5_conv1, shape=[14, 14, 512]), perm=[2, 0, 1])
    block5_conv2 = tf.transpose(tf.reshape(block5_conv2, shape=[14, 14, 512]), perm=[2, 0, 1])
    block5_conv3 = tf.transpose(tf.reshape(block5_conv3, shape=[14, 14, 512]), perm=[2, 0, 1])

    return block5_conv1, block5_conv2, block5_conv3


def get_content_loss(A, B):
    print("exec get_content_loss")
    content_loss = 0.
    for i in range(0, len(A) - 1):
        content_loss += content_loss_layer(A[i], B[i])
    print("content loss : ",content_loss)
    return content_loss


def gram(f1, f2):
    return tf.multiply(f1, f2)


def gram_layer(l1, l2):
    l1_trans = tf.reshape(l1,shape=[512,14*14])
    l2_trans = tf.reshape(l2,shape=[512,14*14])
    g_1 =tf.reduce_sum(tf.matmul(l1_trans,tf.transpose(l2_trans,perm=[1,0])))
    return g_1


def get_style_loss(A, B):
    print("exec get_style_loss")
    style_loss = 0.
    for i in range(0, 1):
        style_loss += gram_layer(A[i], B[i])
    print("content loss : ", style_loss)
    return style_loss


def get_loss():
    print("exec get_loss")
    predict_origin = predict(input_origin)
    predict_middle = predict(input_middle)
    predict_style = predict(input_style)
    global afa_initial,beta_initial
    loss = afa_initial * get_content_loss(predict_origin, predict_middle) + beta_initial * get_style_loss(predict_style,
                                                                                                          predict_middle)
    return loss


with tf.Session() as sess:
    K.set_session(sess)
    model = VGG16(weights='imagenet')
    afa_initial = tf.Variable(np.random.rand())
    beta_initial = tf.Variable(np.random.rand())
    block5_conv1_model = Model(inputs=model.input, outputs=model.get_layer("block5_conv1").output)
    block5_conv2_model = Model(inputs=model.input, outputs=model.get_layer("block5_conv2").output)
    block5_conv3_model = Model(inputs=model.input, outputs=model.get_layer("block5_conv3").output)

    input_origin = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name="input_origin")
    input_style = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name="input_style")

    w_t1 = weight_variable(shape=[224 * 224 * 3])
    b_t1 = tf.Variable(tf.zeros(shape=[224 * 224 * 3], dtype=tf.float32))
    out_1 = tf.multiply(tf.reshape(input_origin, shape=[224 * 224 * 3]), w_t1)
    out_1 = tf.nn.relu(tf.add(out_1, b_t1))

    w_t2 = weight_variable(shape=[224 * 224 * 3])
    b_t2 = tf.Variable(tf.zeros(shape=[224 * 224 * 3], dtype=tf.float32))
    out_2 = tf.multiply(tf.reshape(out_1, shape=[224 * 224 * 3]), w_t2)
    out_2 = tf.nn.relu(tf.add(out_2, b_t2))
    input_middle = tf.reshape(out_2, shape=[1, 224, 224, 3])
    loss = get_loss()
    print("exec get_loss finished")
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    sess.run(tf.global_variables_initializer())
    for step in range(0, 1000):
        feed_dict = {
            input_origin: reader.load_img(r"./data/dog.jpg"),
            input_style: reader.load_img(r"./data/style.jpg")
        }
        print(step)
        train_step.run(feed_dict=feed_dict)
        print(sess.run(loss, feed_dict=feed_dict))
