import tensorflow as tf
from PIL import Image
import numpy as np

saver = tf.train.import_meta_graph("./model/data.chkp.meta")
im = Image.open(r'C:\Users\31826\Desktop\王振薈\03_python\tensorflow_study\dog.png')
input_image = np.array(im).reshape((1, 32 * 32, 3)) / 256
input_image[input_image > 0] = 1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './model/data.chkp')
    # print("number is ", rst.eval(feed_dict={x: input_image, keep_prob: 0.5}))
    predict = tf.get_default_graph().get_tensor_by_name("rst:0")
    print(predict.eval(feed_dict={'x:0': input_image, 'keep_prob:0': 1.}))
