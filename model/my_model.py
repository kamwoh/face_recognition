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
    margin = tf.constant(1.)

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
                                                                      initializer=tf.glorot_uniform_initializer(132),
                                                                      regularizer=slim.l2_regularizer(0.0005),
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


def get_model(lr, n_classes):
    layers = {}
    model_variables = {}
    train_ops = {}

    layers['l_input'] = tf.placeholder(dtype=tf.float32,
                                       shape=[None, 112, 112, 3])

    layers['r_input'] = tf.placeholder(dtype=tf.float32,
                                       shape=[None, 112, 112, 3])

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
                            padding='SAME'):
            lr_conv2d('input', 3, 32, 7, 1, 'conv1', model_variables, layers)  # 224x224x32
            lr_maxpool2d('conv1', 2, 2, 'pool1', layers)  # 112x112x32

            lr_conv2d('pool1', 32, 32, 5, 1, 'conv2', model_variables, layers)  # 112x112x32
            lr_conv2d('conv2', 32, 64, 5, 1, 'conv3', model_variables, layers)  # 112x112x64
            lr_conv2d('conv3', 64, 64, 5, 1, 'conv4', model_variables, layers)  # 112x112x64
            lr_maxpool2d('conv4', 2, 2, 'pool4', layers)  # 56x56x64

            lr_conv2d('pool4', 64, 64, 3, 1, 'conv5', model_variables, layers)  # 56x56x64
            lr_maxpool2d('conv5', 2, 2, 'pool5', layers)  # 28x28x64

            lr_conv2d('pool5', 64, 128, 3, 1, 'conv6', model_variables, layers)  # 28x28x128
            lr_maxpool2d('conv6', 2, 2, 'pool6', layers)  # 14x14x128

            lr_flatten('pool6', 'flat_pool6', layers)  # 25088
            lr_fc('flat_pool6', [6272], 512, 'fc7', model_variables, layers)

            # identification loss (minimize cross entropy)
            ident_w = dict(name='weights', shape=[512, n_classes])
            ident_b = dict(name='biases', shape=[n_classes])

            create_variables(ident_w, ident_b, 'ident_fc', model_variables)

            layers['l_ident_fc'] = slim.fully_connected(layers['l_fc7'],
                                                        n_classes,
                                                        activation_fn=tf.identity,
                                                        scope='ident_fc',
                                                        reuse=True)
            layers['r_ident_fc'] = slim.fully_connected(layers['r_fc7'],
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
            layers['ident_loss'] = tf.reduce_mean(layers['l_ident_loss'] + layers['r_ident_loss'])

            # verification loss (minimize L2 distance between two vector)
            layers['l2_dist'] = l2_distance(layers['l_fc7'], layers['r_fc7'])  # batch_size x 1
            layers['verif_loss'], margin_var = l2_loss(layers['l2_dist'], layers['verif_labels'], 0.5)

            # verification accuracy is l2_dist <= margin_var as mentioned in the paper
            identical_face_bool = layers['l2_dist'] <= margin_var  # batch_size x 1
            identical_face_int = tf.cast(identical_face_bool, tf.int32)
            identical_face_int = tf.reshape(identical_face_int, (-1, 1))
            label_vs_identical = tf.equal(layers['verif_labels'], identical_face_int)
            l_v_i_float = tf.cast(label_vs_identical, tf.float32)

            layers['verif_accuracy'] = tf.reduce_mean(l_v_i_float)

            ident_optimizer = tf.train.AdamOptimizer(lr, name='adam_ident')
            verif_optimizer = tf.train.AdamOptimizer(lr, name='adam_verif')

            # only softmax layer parameters
            ident_op = ident_optimizer.minimize(layers['ident_loss'], var_list=[model_variables.pop('ident_fc/weights'),
                                                                                model_variables.pop('ident_fc/biases')])

            # only the margin
            verif_op = verif_optimizer.minimize(layers['verif_loss'])

            train_ops['ident'] = ident_op
            train_ops['verif'] = verif_op

            for v in tf.global_variables():
                print v

    return layers, train_ops
