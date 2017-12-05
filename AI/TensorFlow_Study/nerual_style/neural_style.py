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
CONTENT_RATE = 1
STYLE_RATE = 1
LEARNING_RATE = 0.5
STEP_SUM = 20000


def get_vgg19_model(input_content, input_style, input_middle):
    """
    create vgg19 model
    :param input_content: data of content image
    :param input_style: data of style image
    :param input_middle: data of generated image
    :return: vgg19 model
    """
    vgg19_content = vgg19_util.get_vgg19(input_content)
    vgg19_style = vgg19_util.get_vgg19(input_style)
    vgg19_middle = vgg19_util.get_vgg19(input_middle)
    return vgg19_content, vgg19_style, vgg19_middle


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
    input_middle = tf.Variable(tf.random_uniform(shape=input_content.shape, dtype=tf.float32, maxval=128, minval=-128),
                               name="weight")

    # get vgg19 model with each input data
    vgg19_content, vgg19_style, vgg19_middle = get_vgg19_model(input_content, input_style, input_middle)

    style_loss_sum = tf.Variable(0.0, trainable=False, name="style_loss_sum")
    content_loss_sum = tf.Variable(0.0, trainable=False, name="content_loss_sum")

    # compute content loss
    for name in CONTENT_LAYERS:
        content_feature = vgg19_content[name]
        middle_feature = vgg19_middle[name]
        content_loss = loss_util.get_content_loss(middle_feature, content_feature) * CONTENT_RATE
        content_loss_sum = tf.add(content_loss_sum, content_loss, name="content_loss_sum")

    # compute content style
    for name in STYLE_LAYERS:
        style_feature = vgg19_style[name]
        middle_feature = vgg19_middle[name]
        style_loss = loss_util.get_style_loss(middle_feature, style_feature) * STYLE_RATE
        style_loss_sum = tf.add(style_loss_sum, style_loss, name="style_loss_sum")

    # compute the total loss
    with tf.name_scope("loss"):
        loss = tf.add(content_loss_sum, style_loss_sum, name="loss")
        tf.summary.scalar("loss", loss)

    # use decayed learning rate
    global_step = tf.Variable(0)
    decayed_learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE, global_step=global_step,
                                                       decay_rate=0.95, staircase=True, decay_steps=100)

    # training step
    train_step = tf.train.AdamOptimizer(decayed_learning_rate).minimize(loss=loss, var_list=[input_middle])

    # model saver
    saver = tf.train.Saver()

    # logger
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./log', session.graph)

    # variable initialization
    session.run(tf.global_variables_initializer())

    # training loop
    for step in range(0, STEP_SUM):
        if step % 10 == 0:
            loss_value, content_loss_value, style_loss_value, input_value = session.run(
                [loss, content_loss_sum, style_loss_sum, input_middle])
            print("step : %i content loss : %g style loss : %g total loss : %g" % (
                step, content_loss_value, style_loss_value, loss_value))
            if step % 100 == 0:
                summary = session.run(merged)
                writer.add_summary(summary, step)
                image_util.save_image(input_value, "./output", "%i.jpg" % step)
                # save model
                saver.save(session, "./model/model.chkp", global_step=step)
        session.run(train_step, feed_dict={global_step: step})

    # close resource
    session.close()
    writer.close()


def transfer(content_image_path, style_image_path):
    tf.set_random_seed(1)
    input_content = image_util.load_img(content_image_path)
    input_style = image_util.load_img(style_image_path)
    train(input_content, input_style)


def main():
    transfer(content_image_path=r"./input/building.jpg", style_image_path=r"./input/star.jpg")


if __name__ == '__main__':
    main()
