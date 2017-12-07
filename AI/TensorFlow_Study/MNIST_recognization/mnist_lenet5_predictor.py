import tensorflow as tf
from PIL import Image
import numpy as np


def predict(file_path):
    """
    use trained model to recognize the number
    :param file_path: picture of number
    :return:number
    """
    saver = tf.train.import_meta_graph("./data.chkp.meta")
    im = Image.open(file_path).convert('L')
    input_image = 255 - np.array(im).reshape((1, 28 * 28))
    input_image[input_image > 0] = 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './data.chkp')
        result = tf.get_default_graph().get_tensor_by_name("rst:0")
        print(result.eval(feed_dict={'x:0': input_image, 'keep_prob:0': 1.}))


def main():
    predict(r'')


if __name__ == "__main__":
    main()
