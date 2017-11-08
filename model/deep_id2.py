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


def get_model(lr, n_classes):
    layers = {}
    model_variables = {}
    train_ops = {}

    layers['l_input'] = tf.placeholder(dtype=tf.float32,
                                       shape=[None, 55, 47, 3])

    layers['r_input'] = tf.placeholder(dtype=tf.float32,
                                       shape=[None, 55, 47, 3])

    layers['verif_labels'] = tf.placeholder(dtype=tf.int32,
                                            shape=[None, 1])

    layers['l_labels'] = tf.placeholder(dtype=tf.int32,
                                        shape=[None, 1])
    layers['r_labels'] = tf.placeholder(dtype=tf.int32,
                                        shape=[None, 1])

    # inference
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu):
        with slim.arg_scope([slim.conv2d],
                            padding='VALID'):
            with slim.arg_scope([slim.model_variable],
                                initializer=tf.glorot_uniform_initializer(),
                                regularizer=slim.l2_regularizer(0.0005)):
                lr_conv2d('input', 3, 20, 4, 1, 'conv1', model_variables, layers)  # 52x44x20
                lr_maxpool2d('conv1', 2, 2, 'pool1', layers)  # 26x22x20

                lr_conv2d('pool1', 20, 40, 3, 1, 'conv2', model_variables, layers)  # 24x20x40
                lr_maxpool2d('conv2', 2, 2, 'pool2', layers)  # 12x10x40

                lr_conv2d('pool2', 40, 60, 3, 1, 'conv3', model_variables, layers)  # 10x8x60
                lr_maxpool2d('conv3', 2, 2, 'pool3', layers)  # 5x4x60

                lr_local_conv2d('pool3', [5, 4, 60], 80, 2, 1, 'conv4', model_variables, layers)  # 4x3x80

                lr_flatten('pool3', 'flat_pool3', layers)  # 1200
                lr_flatten('conv4', 'flat_conv4', layers)  # 960

                lr_concat(['flat_pool3', 'flat_conv4'], 1, 'concat_deepid2', layers)  # 2160
                lr_fc('concat_deepid2', [2160], 160, 'deepid2', model_variables, layers)  # 160

                # identification loss (minimize cross entropy)
                ident_w = dict(name='weights', shape=[160, n_classes])
                ident_b = dict(name='biases', shape=[n_classes])

                create_variables(ident_w, ident_b, 'ident_fc', model_variables)

                layers['l_ident_fc'] = slim.fully_connected(layers['l_deepid2'],
                                                            n_classes,
                                                            activation_fn=tf.identity,
                                                            scope='ident_fc',
                                                            reuse=True)
                layers['r_ident_fc'] = slim.fully_connected(layers['r_deepid2'],
                                                            n_classes,
                                                            activation_fn=tf.identity,
                                                            scope='ident_fc',
                                                            reuse=True)

                layers['l_ident_softmax'] = slim.softmax(layers['l_ident_fc'])
                layers['r_ident_softmax'] = slim.softmax(layers['r_ident_fc'])

                layers['l_ident_loss'] = tf.losses.sparse_softmax_cross_entropy(logits=layers['l_ident_fc'],
                                                                                labels=layers['l_labels'])
                layers['r_ident_loss'] = tf.losses.sparse_softmax_cross_entropy(logits=layers['r_ident_fc'],
                                                                                labels=layers['r_labels'])
                layers['ident_loss'] = layers['l_ident_loss'] + layers['r_ident_loss']

                # verification loss (minimize L2 distance between two deepid vector)
                # lambda is a hyperparameter which is 0.1 produce best result indicated in the paper
                layers['l2_dist'] = l2_distance(layers['l_deepid2'], layers['r_deepid2'])  # batch_size x 1
                layers['verif_loss'], margin_var = l2_loss(layers['l2_dist'], layers['verif_labels'], 0.1)

                # verification accuracy is l2_dist <= margin_var as mentioned in the paper
                # identical_face_bool = layers['l2_dist'] <= margin_var
                # identical_face_int = tf.cast(identical_face_bool, tf.int32)
                # identical_face_int = tf.reshape(identical_face_int, (-1, 1))
                # label_vs_identical = tf.equal(layers['verif_labels'], identical_face_int)
                # l_v_i_float = tf.cast(label_vs_identical, tf.float32)

                compare = tf.argmax(layers['l_ident_fc'], axis=1)

                layers['verif_accuracy'] = tf.reduce_mean(accuracy)

                layers['l_loss'] = layers['l_ident_loss'] + layers['verif_loss']
                layers['r_loss'] = layers['r_ident_loss'] + layers['verif_loss']

                # feature loss reduce by l_loss and r_loss
                layers['feature_loss'] = layers['l_loss'] * layers['l_deepid2'] + layers['r_loss'] * layers['r_deepid2']

                ident_optimizer = tf.train.AdamOptimizer(lr, name='adam_ident')
                verif_optimizer = tf.train.AdamOptimizer(lr, name='adam_verif')
                feature_optimizer = tf.train.AdamOptimizer(lr, name='adam_feature')

                # only softmax layer parameters
                ident_op = ident_optimizer.minimize(layers['ident_loss'],
                                                    var_list=[model_variables.pop('ident_fc/weights'),
                                                              model_variables.pop('ident_fc/biases')])
                # only the margin
                verif_op = verif_optimizer.minimize(layers['verif_loss'], var_list=[margin_var])

                # conv net
                feature_op = feature_optimizer.minimize(layers['feature_loss'], var_list=model_variables.values())

                train_ops['ident'] = ident_op
                train_ops['verif'] = verif_op
                train_ops['feature'] = feature_op

    return layers, train_ops
