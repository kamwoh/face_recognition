import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from dataset import sklearn_dataset as sk

dirname = os.path.dirname(__file__)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    subjectName = 'face-recog-1510565976.19'
    imagePath = '/home/woh/Dataset/Abdullah_Gul/2.jpeg'
    loadPath = '{}/trained/{}/{}-{}'.format(dirname, subjectName, subjectName, 31201)
    saver = tf.train.import_meta_graph(loadPath + '.meta')
    saver.restore(sess, loadPath)

    # print [n.name for n in tf.get_default_graph().as_graph_def().node]
    for v in tf.global_variables():
        print(v)

    softmax = tf.get_default_graph().get_tensor_by_name('softmax/Reshape_1:0')
    inputs = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    is_training = tf.get_default_graph().get_tensor_by_name('Placeholder_2:0')
    deepid = tf.get_default_graph().get_tensor_by_name('fully_connected/Relu:0')
    keep_prob = tf.get_default_graph().get_tensor_by_name('Placeholder_3:0')
    argmax = tf.get_default_graph().get_tensor_by_name('ArgMax_1:0')
    ident_fc = tf.get_default_graph().get_tensor_by_name('fully_connected_1/Relu:0')

    data = np.load('{}/dataset/lfw/image_mean_and_std.npz'.format(dirname))
    mean = data['channelMean']
    std = data['std']
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (47, 55), cv2.INTER_CUBIC)
    imageCopy = np.copy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image -= mean
    image /= std + 1e-7
    image = np.expand_dims(image, 0)

    pred = sess.run(softmax, feed_dict={
        inputs: image,
        is_training: False,
        keep_prob: 1
    })

    # print 'pred', pred
    # print 'sorted', np.argsort(pred[0])[::-1]
    # print sk.mapTargetToName('lfwColor_0.5' , pred[0])

    indices = np.argsort(pred[0])[::-1]
    for j in range(10):
        print(sk.mapTargetToName('lfwColor_0.5', indices[j]), pred[0][indices[j]])

    cv2.imshow('test', imageCopy)
    cv2.waitKey(0)
