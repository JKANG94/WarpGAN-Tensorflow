import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, use_bias=True, scope='deconv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                       strides=stride, padding='SAME', use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias)
            x = relu(x)
            x = batch_norm(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias)
            x = batch_norm(x)

        return x + x_init


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def batch_norm(x, scope = 'batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        center=True, scale=True,
                                        scope=scope)
##################################################################################
# Loss function
##################################################################################

def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge' :
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss

def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss

def classification_loss(logit, label) :
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

    return loss

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def total_variation(images, name=None):

    ndims = images.get_shape().ndims
    if ndims == 3:

      pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
      pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
      sum_axis = None

    elif ndims == 4:
      pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]

      pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]

      sum_axis = [1, 2, 3]

    else:

      raise ValueError('\'images\' must be either 3 or 4-dimensional.')

    tot_var = tf.reduce_mean(tf.square(pixel_dif1), axis=sum_axis) +
        tf.reduce_mean(tf.square(pixel_dif2), axis=sum_axis)

  return tot_var

def binary_label_loss(hat_c, hat_cls):
    _, r = hat_c.shape
    h = r / (np.linalg.norm(hat_c, ord = 1, axis = 1) + 0.001)
    B = np.zeros(hat_c.shape, dtype = 'float32')

    bar_c = tf.where(tf.equal(hat_c, -1), B, hat_c) # label operator
    Loss = tf.reduce_mean(h * tf.reduce_sum(tf.abs(hat_c) * tf.nn.sigmoid_cross_entropy_with_logits(labels=bar_c, logits=hat_cls), axis = 1))
    return Loss

