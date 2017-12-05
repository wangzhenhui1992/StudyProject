# -*- utf-8 -*-
"""neural_style can be used to apply the style of one picture to another
"""
import tensorflow as tf
from tensorflow.python.keras import backend
from CNN.neural_style import vgg19_util
from CNN.neural_style import loss_util
from CNN.neural_style import image_util

CONTENT_LAYERS = ['block5_conv2']
STYLE_LAYERS = ["block1_conv2", "block2_conv2", "block3_conv2", "block4_conv2"]
CONTENT_RATE = 1.
STYLE_RATE = 1
LEARNING_RATE = 0.5
STEP_SUM = 20000


def get_feature(layer_name, input_content_or_style, middle, session):
    """
    Get feature of the previous layer
    :param layer_name: Name of the previous layer
    :param input_content_or_style: Data of the content image or the style image
    :param middle:Data of generated image
    :param session: Tensorflow session
    :return: Features
    """

    content_or_style_feature = vgg19_util.get_vgg19(input_content_or_style, layer_name)
    middle_feature_output = vgg19_util.get_vgg19(middle, layer_name)
    return content_or_style_feature, middle_feature_output


def train(input_content, input_style):
    """
    Training method
    :param input_content: Data of content image
    :param input_style: Data of style image
    :return: None
    """

    session = tf.Session()

    backend.set_session(session)

    # generate an image by random
    input_init = tf.random_uniform(shape=input_content.shape, dtype=tf.float32, maxval=128, minval=-128)
    input_middle = tf.Variable(input_init, name="weight")

    with tf.name_scope("loss"):
        style_loss_sum = tf.Variable(0.0, trainable=False, name="style_loss_sum")
        content_loss_sum = tf.Variable(0.0, trainable=False, name="content_loss_sum")
        tf.summary.scalar("style_loss_sum", style_loss_sum)
        tf.summary.scalar("content_loss_sum", content_loss_sum)

    # compute content loss
    for name in CONTENT_LAYERS:
        content_feature, middle_feature = get_feature(name, input_content, input_middle, session)
        content_loss = loss_util.get_content_loss(middle_feature, content_feature) * CONTENT_RATE
        with tf.name_scope("loss"):
            content_loss_sum = tf.add(content_loss_sum, content_loss, name="content_loss_sum")

    # compute content style
    for name in STYLE_LAYERS:
        style_feature, middle_feature = get_feature(name, input_style, input_middle, session)
        style_loss = loss_util.get_style_loss(middle_feature, style_feature) * STYLE_RATE
        with tf.name_scope("loss"):
            style_loss_sum = tf.add(style_loss_sum, style_loss, name="style_loss_sum")
    with tf.name_scope("loss"):
        loss = tf.add(content_loss_sum, style_loss_sum, name="loss")
        tf.summary.scalar("loss", loss)
    global_step = tf.Variable(0)
    decayed_learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE, global_step=global_step,
                                                       decay_rate=0.95, staircase=True, decay_steps=100)
    train_step = tf.train.AdamOptimizer(decayed_learning_rate).minimize(loss=loss, var_list=[input_middle])
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./log', session.graph)
    session.run(tf.global_variables_initializer())
    for step in range(0, STEP_SUM):
        _, summary = session.run([train_step, merged], feed_dict={global_step: step})
        writer.add_summary(summary, step)
        if step % 10 == 0:
            loss_value, content_loss_value, style_loss_value, input_value = session.run(
                [loss, content_loss_sum, style_loss_sum, input_middle])
            print("step : %i content loss : %g style loss : %g total loss : %g" % (
                step, content_loss_value, style_loss_value, loss_value))
            if step % 100 == 0:
                image_util.save_image(input_value, "./output", "%i.jpg" % step)
                # save model
                saver.save(session, "./model/model.chkp", global_step=step)
    session.close()


def transfer(content_image_path, style_image_path):
    input_content = image_util.load_img(content_image_path)
    input_style = image_util.load_img(style_image_path)
    train(input_content, input_style)


def main():
    transfer(content_image_path=r"./input/building.jpg", style_image_path=r"./input/star.jpg")


if __name__ == '__main__':
    main()
