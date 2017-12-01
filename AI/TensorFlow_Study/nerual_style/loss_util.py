# -*- utf-8 -*-
"""
util to compute losses
"""

import tensorflow as tf


def get_content_loss(middle_feature, content_feature):
    """
    compute the content loss
    :param middle_feature: features of generated image
    :param content_feature: features of content image
    :return: content loss
    """
    #content_loss_out = tf.reduce_sum(tf.square(middle_feature - content_feature)) / 2
    content_loss_out = tf.nn.l2_loss(middle_feature - content_feature)
    return content_loss_out


def get_style_loss(middle_feature, style_feature):
    """
    compute the style loss
    :param middle_feature: feature of generated image
    :param style_feature: feature of style image
    :return: style loss
    """
    _, height, width, channel = map(lambda element: element.value, style_feature.get_shape())
    size = height * width * channel
    middle_gram = gram(middle_feature, channel)
    style_gram = gram(style_feature, channel)
    # style_loss = tf.reduce_sum(tf.square(middle_gram - style_gram)) / 4 / size / size
    style_loss = tf.nn.l2_loss(middle_gram-style_gram) / 2. / size / size /3.
    return style_loss


def gram(feature, channel):
    """
    compute gram metrix
    :param feature: features
    :param channel: channel of features
    :return:gram metrix
    """
    feature_reshape = tf.reshape(feature, shape=[-1, channel])
    feature_gram = tf.matmul(tf.transpose(feature_reshape), feature_reshape)
    return feature_gram
