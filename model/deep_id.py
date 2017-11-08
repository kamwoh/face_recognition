import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib import keras


def l2_loss(l2_dist, labels, l):
    """
    refer to https://stackoverflow.com/questions/37479119/doing-pairwise-distance-computation-with-tensorflow

    verification loss function

    loss = verif(l_deepid2, r_deepid2, labels, parameters)
    if labels == 1:
        loss = 1/2 * l2diff
    else:
        loss = 1/2 * max (0, margin - l2diff)

    """
    # margin parameters to learn from L2 loss function
    margin = tf.Variable(1., name='margin')

    labels = tf.to_float(labels)

    match_loss = tf.square(l2_dist)
    mismatch_loss = tf.maximum(0., tf.subtract(margin, tf.square(l2_dist)))

    # if label is 1, only match_loss will count, otherwise mismatch_loss
    loss = tf.add(tf.multiply(labels, match_loss),
                  tf.multiply((1 - labels), mismatch_loss))

    loss_mean = tf.reduce_mean(loss)
    return l * loss_mean, margin


def l2_distance(l_deepid2, r_deepid2):
    # Euclidean distance between l_deepid2,r_deepid2
    l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(l_deepid2, r_deepid2)),
                                   reduction_indices=1))

    # referring paper's verification loss
    l2diff = tf.divide(l2diff, 2)

    return l2diff


def create_variables(w, b, scope, model_variables):
    model_variables['{}/weights'.format(scope)] = slim.model_variable('{}/{}'.format(scope, w['name']),
                                                                      shape=w['shape'])

    model_variables['{}/biases'.format(scope)] = slim.model_variable('{}/{}'.format(scope, b['name']),
                                                                     initializer=tf.constant_initializer(0.1),
                                                                     shape=b['shape'])


def lr_local_conv2d(input_scope, input_shape, num_outputs, kernel_size, stride, scope, model_variables, layers):
    n_stride_y = (input_shape[0] - kernel_size) / stride + 1
    n_stride_x = (input_shape[1] - kernel_size) / stride + 1

    w = dict(name='kernel', shape=[n_stride_y * n_stride_x,
                                   kernel_size * kernel_size * input_shape[2],
                                   num_outputs])
    b = dict(name='bias', shape=[n_stride_y,
                                 n_stride_x,
                                 num_outputs])

    create_variables(w, b, scope, model_variables)

    layers['l_{}'.format(scope)] = keras.layers.LocallyConnected2D(num_outputs,
                                                                   kernel_size,
                                                                   (stride, stride),
                                                                   activation=tf.nn.relu,
                                                                   name=scope,
                                                                   reuse=True)(layers['l_{}'.format(input_scope)])
    layers['r_{}'.format(scope)] = keras.layers.LocallyConnected2D(num_outputs,
                                                                   kernel_size,
                                                                   (stride, stride),
                                                                   activation=tf.nn.relu,
                                                                   name=scope,
                                                                   reuse=True)(layers['r_{}'.format(input_scope)])


def lr_conv2d(input_scope, num_inputs, num_outputs, kernel_size, stride, scope, model_variables, layers):
    w = dict(name='weights', shape=[kernel_size, kernel_size, num_inputs, num_outputs])
    b = dict(name='biases', shape=[num_outputs])

    create_variables(w, b, scope, model_variables)

    layers['l_{}'.format(scope)] = slim.conv2d(layers['l_{}'.format(input_scope)],
                                               num_outputs,
                                               kernel_size,
                                               stride,
                                               scope=scope,
                                               reuse=True)
    layers['r_{}'.format(scope)] = slim.conv2d(layers['r_{}'.format(input_scope)],
                                               num_outputs,
                                               kernel_size,
                                               stride,
                                               scope=scope,
                                               reuse=True)


def lr_maxpool2d(input_scope, kernel_size, stride, scope, layers):
    layers['l_{}'.format(scope)] = slim.max_pool2d(layers['l_{}'.format(input_scope)], kernel_size, stride, scope=scope)
    layers['r_{}'.format(scope)] = slim.max_pool2d(layers['r_{}'.format(input_scope)], kernel_size, stride, scope=scope)


def lr_flatten(input_scope, scope, layers):
    layers['l_{}'.format(scope)] = slim.flatten(layers['l_{}'.format(input_scope)], scope=scope)
    layers['r_{}'.format(scope)] = slim.flatten(layers['r_{}'.format(input_scope)], scope=scope)


def lr_concat(input_scopes, axis, scope, layers):
    l_concat = []
    r_concat = []

    for input_scope in input_scopes:
        l_concat.append(layers['l_{}'.format(input_scope)])
        r_concat.append(layers['r_{}'.format(input_scope)])

    layers['l_{}'.format(scope)] = tf.concat(l_concat, 1, name=scope)
    layers['r_{}'.format(scope)] = tf.concat(r_concat, 1, name=scope)


def lr_fc(input_scope, input_shape, num_outputs, scope, model_variables, layers):
    w = dict(name='weights', shape=[reduce(lambda x, y: x * y, input_shape, 1), num_outputs])
    b = dict(name='biases', shape=[num_outputs])

    create_variables(w, b, scope, model_variables)

    layers['l_{}'.format(scope)] = slim.fully_connected(layers['l_{}'.format(input_scope)],
                                                        num_outputs,
                                                        scope=scope,
                                                        reuse=True)
    layers['r_{}'.format(scope)] = slim.fully_connected(layers['r_{}'.format(input_scope)],
                                                        num_outputs,
                                                        scope=scope,
                                                        reuse=True)


def lr_batchnorm(input_scope, scope, layers):
    layers['l_{}'.format(scope)] = slim.batch_norm(layers['l_{}'.format(input_scope)],
                                                   scope='l_{}'.format(scope))


def get_model(lr, n_classes):
    layers = {}
    train_ops = {}

    layers['inputs'] = tf.placeholder(dtype=tf.float32,
                                      shape=[None, 55, 47, 3])

    layers['zero_mean'] = tf.map_fn(lambda i: tf.image.per_image_standardization(i), layers['inputs'])
    layers['labels'] = tf.placeholder(dtype=tf.int64,
                                      shape=[None, 1])
    layers['is_training'] = tf.placeholder(dtype=tf.bool)

    # inference
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.glorot_uniform_initializer(seed=132),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        activation_fn=tf.nn.relu):
        with slim.arg_scope([slim.conv2d],
                            padding='VALID'):
            layers['conv1'] = slim.conv2d(layers['zero_mean'], 20, 4, 1)
            layers['bn1'] = slim.batch_norm(layers['conv1'], is_training=layers['is_training'])
            layers['pool1'] = slim.max_pool2d(layers['bn1'], 2, 2)

            layers['conv2'] = slim.conv2d(layers['pool1'], 40, 3, 1)
            layers['pool2'] = slim.max_pool2d(layers['conv2'], 2, 2)

            layers['conv3'] = slim.conv2d(layers['pool2'], 60, 3, 1)
            layers['pool3'] = slim.max_pool2d(layers['conv3'], 2, 2)

            layers['conv4'] = keras.layers.LocallyConnected2D(80, 2, 1, activation=tf.nn.relu)(layers['pool3'])
            layers['flatten_pool3'] = slim.flatten(layers['pool3'])
            layers['flatten_conv4'] = slim.flatten(layers['conv4'])

            layers['concat'] = tf.concat([layers['flatten_conv4'], layers['flatten_pool3']], 1)
            layers['deepid_fc'] = slim.fully_connected(layers['concat'], 160)
            layers['deepid_dropout'] = slim.dropout(layers['deepid_fc'], 0.5, is_training=layers['is_training'])

            layers['ident_fc'] = slim.fully_connected(layers['deepid_fc'], n_classes, activation_fn=tf.identity)

            layers['argmax'] = tf.expand_dims(tf.argmax(layers['ident_fc'], axis=1), 1)
            layers['equal'] = tf.equal(layers['argmax'], layers['labels'])
            layers['cast'] = tf.cast(layers['equal'], tf.float32)

            layers['accuracy'] = tf.reduce_mean(layers['cast'])
            layers['loss'] = tf.losses.sparse_softmax_cross_entropy(labels=layers['labels'],
                                                                    logits=layers['ident_fc'])

            ident_optimizer = tf.train.AdamOptimizer(lr)

            ident_op = ident_optimizer.minimize(layers['loss'])

            train_ops['ident'] = ident_op

    return layers, train_ops
