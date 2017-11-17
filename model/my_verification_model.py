import tensorflow as tf

from tensorflow.contrib import slim


def get_model(lr, global_step=None):
    layers = {}
    train_ops = {}
    # todo: stack inputs
    layers['l_inputs'] = tf.placeholder(tf.float32, shape=[None, 160], name='l_inputs')
    layers['r_inputs'] = tf.placeholder(tf.float32, shape=[None, 160], name='r_inputs')
    layers['verif_labels'] = tf.placeholder(tf.int64, shape=[None, ], name='verif_labels')
    layers['keep_prob'] = tf.placeholder(tf.float32, name='keep_prob')
    layers['is_training'] = tf.placeholder(tf.bool, name='is_training')

    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        biases_initializer=tf.constant_initializer(0.1),
                        activation_fn=tf.nn.relu):
        layers['l_local_fc'] = slim.fully_connected(layers['l_inputs'], 80, scope='l_local_fc')
        layers['r_local_fc'] = slim.fully_connected(layers['r_inputs'], 80, scope='r_local_fc')
        layers['concat_local_fc'] = tf.concat([layers['l_local_fc'], layers['r_local_fc']], 1,
                                              name='concat_local_fc')

        layers['fc_1'] = slim.fully_connected(layers['concat_local_fc'], 4800, scope='fc_1')
        layers['fc_1_dropout'] = slim.dropout(layers['fc_1'], keep_prob=layers['keep_prob'],
                                              is_training=layers['is_training'], scope='fc_1_dropout')

        layers['output_fc'] = slim.fully_connected(layers['fc_1_dropout'], 2, scope='output', activation_fn=tf.identity)
        layers['output_softmax'] = slim.softmax(layers['output_fc'], scope='output_softmax')
        layers['softmax_loss'] = tf.losses.sparse_softmax_cross_entropy(labels=layers['verif_labels'],
                                                                        logits=layers['output_fc'])

        layers['argmax'] = tf.argmax(layers['output_fc'], axis=1, name='argmax')
        layers['equal'] = tf.equal(layers['argmax'], layers['verif_labels'], name='equal')
        layers['cast'] = tf.cast(layers['equal'], dtype=tf.float32, name='cast')
        layers['accuracy'] = tf.reduce_mean(layers['cast'], name='accuracy')

        verif_optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        verif_op = verif_optimizer.minimize(layers['softmax_loss'], global_step=global_step)

        train_ops['verif_op'] = verif_op

    return layers, train_ops
