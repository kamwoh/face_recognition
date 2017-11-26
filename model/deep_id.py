import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib import keras

def leaky_relu(alpha):
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
    return op

def get_model(lr, n_classes, global_step=None):
    layers = {}
    train_ops = {}

    layers['inputs'] = tf.placeholder(dtype=tf.float32,
                                      shape=[None, 55, 47, 3])

    # layers['zero_mean'] = tf.map_fn(lambda i: tf.image.per_image_standardization(i), layers['inputs'])
    layers['labels'] = tf.placeholder(dtype=tf.int64,
                                      shape=[None,])
    # layers['onehot_labels'] = tf.one_hot(layers['labels'], n_classes)
    layers['is_training'] = tf.placeholder(dtype=tf.bool)
    layers['keep_prob'] = tf.placeholder(dtype=tf.float32)

    # inference
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        biases_initializer=tf.constant_initializer(0.1),
                        activation_fn=tf.nn.relu):
        with slim.arg_scope([slim.conv2d],
                            padding='VALID'):
            layers['conv1'] = slim.conv2d(layers['inputs'], 20, 4, 1)
            # layers['bn1'] = slim.batch_norm(layers['conv1'], is_training=layers['is_training'])
            layers['pool1'] = slim.max_pool2d(layers['conv1'], 2, 2)

            layers['conv2'] = slim.conv2d(layers['pool1'], 40, 3, 1)
            layers['pool2'] = slim.max_pool2d(layers['conv2'], 2, 2)

            layers['conv3'] = slim.conv2d(layers['pool2'], 60, 3, 1)
            layers['pool3'] = slim.max_pool2d(layers['conv3'], 2, 2)

            layers['conv4'] = keras.layers.LocallyConnected2D(80, 2, 1, activation=tf.nn.relu)(layers['pool3'])
            # layers['conv4'] = slim.conv2d(layers['pool3'], 80, 2, 1)
            layers['flatten_pool3'] = slim.flatten(layers['pool3'])
            layers['flatten_conv4'] = slim.flatten(layers['conv4'])

            layers['concat'] = tf.concat([layers['flatten_conv4'], layers['flatten_pool3']], 1)
            layers['deepid_fc'] = slim.fully_connected(layers['concat'], 160)
            layers['deepid_dropout'] = slim.dropout(layers['deepid_fc'], layers['keep_prob'], is_training=layers['is_training'])

            layers['ident_fc'] = slim.fully_connected(layers['deepid_dropout'], n_classes, activation_fn=tf.identity)
            layers['ident_softmax'] = slim.softmax(layers['ident_fc'])
            # print layers['ident_softmax']
            layers['ident_pred'] = tf.argmax(layers['ident_softmax'], axis=1)

            layers['argmax'] = tf.argmax(layers['ident_fc'], axis=1)
            # print layers['argmax']
            layers['equal'] = tf.equal(layers['argmax'], layers['labels'])
            layers['cast'] = tf.cast(layers['equal'], tf.float32)

            layers['accuracy'] = tf.reduce_mean(layers['cast'])

            layers['loss'] = tf.losses.sparse_softmax_cross_entropy(labels=layers['labels'],
                                                                    logits=layers['ident_fc'])

            layers['softmax_error'] = tf.reduce_mean(tf.cast(tf.not_equal(layers['ident_pred'], layers['labels']), tf.float32))

            # vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # layers['grad'] = tf.gradients(layers['loss'], vars)
            # layers['updates'] = []
            # for v, grad in zip(vars, layers['grad']):
            #     layers['updates'].append(tf.assign_sub(v, lr*grad))
            #
            # train_op = tf.group(*layers['updates'])

            # ident_optimizer = tf.train.RMSPropOptimizer(lr, momentum=0.1)
            # ident_optimizer = tf.train.AdamOptimizer(lr)
            ident_optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
            # ident_optimizer = tf.train.GradientDescentOptimizer(lr)
            ident_op = ident_optimizer.minimize(layers['loss'], global_step=global_step)

            # train_ops['ident'] = train_op
            train_ops['ident'] = ident_op

    return layers, train_ops

# get_model(0.1, 2)