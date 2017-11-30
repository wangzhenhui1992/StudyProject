# -*- utf-8 -*-

from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as backend
import tensorflow as tf
import reader
from PIL import Image
import numpy as np

CONTENT_LAYERS = ['block1_conv2']
STYLE_LAYERS = ['block1_conv2', "block2_conv2", "block3_conv2"]
CONTENT_RATE = 1
STYLE_RATE = 1

"""
def weight_variable(shape):
    initial = tf.random_uniform(maxval=256, minval=0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name="weight")
"""


def get_content_loss(middle_feature_input, origin_feature_input):
    _, height, width, channel = map(lambda i: i.value, origin_feature_input.get_shape())
    size = height * width * channel
    content_loss_out = tf.reduce_sum(tf.square(middle_feature_input - origin_feature_input)) / 2
    return content_loss_out


def get_style_loss(middle_feature_input, style_feature_input):
    _, height, width, channel = map(lambda i: i.value, style_feature_input.get_shape())
    size = height * width * channel
    middle_feature_reshape = tf.reshape(middle_feature_input, shape=[-1, channel])
    style_feature_reshape = tf.reshape(style_feature, shape=[-1, channel])
    middle_gram = tf.matmul(tf.transpose(middle_feature_reshape), middle_feature_reshape)
    style_gram = tf.matmul(tf.transpose(style_feature_reshape), style_feature_reshape)
    style_loss_out = tf.reduce_sum(tf.square(middle_gram - style_gram)) / 4 / size / size
    return style_loss_out


def show_image(image_data, step):
    image_data = image_data.astype(dtype=np.float64)
    image_arr = np.reshape(image_data, [224, 224, 3])
    image_arr[..., 0] += 103.939
    image_arr[..., 1] += 116.779
    image_arr[..., 2] += 123.68
    image_arr = image_arr[..., ::-1]
    image_arr = np.clip(image_arr, 0, 255).astype(dtype=np.uint8)
    # print(image_arr)
    Image.fromarray(image_arr).save("./image/" + str(step) + ".jpg")


def get_non_trainable_model(layer_name, model_origin):
    non_trainable_model = Model(inputs=model_origin.input, outputs=model_origin.get_layer(layer_name).output)
    non_trainable_model.trainable = False
    for layer in non_trainable_model.layers:
        layer.trainable = False
    return non_trainable_model


def get_feature(model_vgg, layer_name, input_origin_or_style, middle):
    origin_or_style_feature = vgg19(input_origin_or_style, layer_name)
    middle_feature_output= vgg19(middle, layer_name)
    return origin_or_style_feature, middle_feature_output


def load_img(path):
    img = image.load_img(path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def vgg19(input, layer_name):
    x = input
    for l in model_vgg.layers:
        if isinstance(l, Conv2D):
            weights = l.weights
            kernel = sess.run(weights[0])
            bias = sess.run(weights[1])
            kernel_tensor = tf.constant(kernel)
            bias_tensor = tf.constant(bias)
            x = tf.nn.conv2d(x, filter=kernel_tensor, padding="SAME", strides=(1, 2, 2, 1))
        elif isinstance(l, MaxPooling2D):
            x = tf.nn.max_pool(x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")
        if l.name == layer_name:
            break
    return x


sess = tf.Session()
backend.set_session(sess)
# read data
input_origin_arr = load_img(r"./data/dog.jpg")  # 1,224,224,3
input_style_arr = load_img(r"./data/star.jpg")  # 1,224,224,3
input_origin = preprocess_input(input_origin_arr)
input_style = preprocess_input(input_style_arr)

input_middle = tf.Variable(tf.random_uniform(shape=input_origin.shape, dtype=tf.float32, maxval=128, minval=-128),
                           name="weight")
# cnn model
model_vgg = VGG19(weights='imagenet')
model_vgg.trainable = False
for l in model_vgg.layers:
    l.trainable = False
#    for l in model_vgg.layers :
#    print(l.name)
style_loss_sum = 0.0
content_loss_sum = 0.0
for name in CONTENT_LAYERS:
    origin_feature, middle_feature = get_feature(model_vgg, name, input_origin, input_middle)
    content_loss = get_content_loss(middle_feature, origin_feature)
    content_loss_sum += content_loss

for name in STYLE_LAYERS:
    style_feature, middle_feature = get_feature(model_vgg, name, input_style, input_middle)
    style_loss = get_style_loss(middle_feature, style_feature)
    style_loss_sum += style_loss
with tf.name_scope("loss"):
    loss = tf.add(content_loss_sum * CONTENT_RATE, style_loss_sum * STYLE_RATE, name="loss")
# loss = c_loss_1 + s_loss_1
train_step = tf.train.AdamOptimizer(0.5).minimize(loss=loss, var_list=[input_middle])
sess.run(tf.global_variables_initializer())
for step in range(0, 5000):
    sess.run(train_step)
    if step % 10 == 0:
        loss_value, content_loss_value, style_loss_value, input_value = sess.run(
            [loss, content_loss_sum, style_loss_sum, input_middle])
        print(step, " content loss : ", content_loss_value, " style loss : ", style_loss_value, " sum : ",
              loss_value)
        if step % 100 == 0:
            show_image(input_value, step)
