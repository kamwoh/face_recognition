import os

import numpy as np
import tensorflow as tf

import graph
from dataset import sklearn_dataset as sk
from dataset import facescrub_dataset as fs
from model import deep_id
import time


def pipeline_deepid():
    # subjectName = 'face-recog-{}'.format(time.time())
    subjectName = 'face-recog-{}'.format('1510879123.02')
    dirname = os.path.dirname(__file__)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    lr = 0.01
    batchSize = 128
    trainEpoch = 30

    trainLFW = False

    if trainLFW:
        key = 'lfwColor_0.5'
        (trainX, trainY), (valX, valY), (testX, testY) = sk.separateToTrainValTest(key, 0.7, 0.1, 0.2)
        trainDataGenerator = sk.getDataGenerator('augmentedDataGenerator',
                                                 X=trainX,
                                                 Y=trainY,
                                                 batchSize=batchSize,
                                                 shuffle=True,
                                                 scaledSize=(55, 47),
                                                 fit=True)
        valDataGenerator = sk.getDataGenerator('defaultDataGenerator',
                                               X=valX,
                                               Y=valY,
                                               batchSize=batchSize,
                                               shuffle=True,
                                               scaledSize=(55, 47),
                                               trainKey='augmentedDataGenerator')
        testDataGenerator = sk.getDataGenerator('defaultDataGenerator',
                                                X=testX,
                                                Y=testY,
                                                batchSize=batchSize,
                                                shuffle=True,
                                                scaledSize=(55, 47),
                                                trainKey='augmentedDataGenerator')
        trainSteps = trainEpoch * (trainX.shape[0] / batchSize)
        testSteps = int(testX.shape[0] / batchSize)
        nClasses = sk.getClasses(key)
    else:
        datadirPrefix = '{}/dataset/facescrub_crop'.format(dirname)
        meanstdDir = '{}/dataset/facescrub_meanstd'.format(dirname)
        nClasses = len(os.listdir(datadirPrefix))
        trainDataDir = datadirPrefix + '_train'
        valDataDir = datadirPrefix + '_val'
        testDataDir = datadirPrefix + '_test'

        trainDataGenerator, nTrainImages = fs.getDataGenerator(trainDataDir, meanstdDir, batchSize, True)
        valDataGenerator, nValImages = fs.getDataGenerator(valDataDir, meanstdDir, batchSize, True)
        testDataGenerator, nTestImages = fs.getDataGenerator(testDataDir, meanstdDir, batchSize, True)

        print 'train -', nTrainImages
        print 'val -', nValImages
        print 'test -', nTestImages
        print 'classes -', nClasses

        trainSteps = trainEpoch * int(nTrainImages / batchSize)
        testSteps = nTestImages / batchSize

    fig1, ax1 = graph.getFigAx()
    fig2, ax2 = graph.getFigAx()

    with tf.Session(config=config) as sess:
        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(lr, globalStep, 100000, 0.9)

        valInterval = 1
        saveInterval = 500
        updateInterval = 100
        saveDir = '{}/trained/{}'.format(dirname, subjectName)

        layers, train_ops = deep_id.get_model(learningRate, nClasses, globalStep)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=None)

        layerFetches = [layers['loss'],
                        layers['accuracy']]
        fetches = layerFetches + train_ops.values()

        # trainLosses = []
        # trainAccuracy = []
        # valLosses = []
        # valAccuracy = []
        trainLosses, trainAccuracy, valLosses, valAccuracy = load('{}/trained/{}'.format(dirname, subjectName),
                                                                  subjectName,
                                                                  42280,
                                                                  saver,
                                                                  sess,
                                                                  globalStep)

        # print 'current learning rate', sess.run(learningRate)

        for step in xrange(trainSteps):
            x, y = trainDataGenerator.next()
            loss, acc = sess.run(fetches, feed_dict={
                layers['inputs']: x,
                layers['labels']: y,
                layers['is_training']: True,
                layers['keep_prob']: 0.5
            })[:2]

            trainLosses.append(loss)
            trainAccuracy.append(acc)

            if step % valInterval == 0:
                x, y = valDataGenerator.next()
                loss, acc = sess.run(layerFetches, feed_dict={
                    layers['inputs']: x,
                    layers['labels']: y,
                    layers['is_training']: False,
                    layers['keep_prob']: 1.0
                })

                valLosses.append(loss)
                valAccuracy.append(acc)

            if step % updateInterval == 0:
                graph.refresh(fig1, ax1, trainLosses[::updateInterval], 'b-')
                graph.refresh(fig1, ax1, trainAccuracy[::updateInterval], 'r-')
                graph.refresh(fig2, ax2, valLosses[::updateInterval], 'b-')
                graph.refresh(fig2, ax2, valAccuracy[::updateInterval], 'r-')

                print '---', step, '---'
                print 'last train', trainLosses[-1], trainAccuracy[-1]
                print 'last validation', valLosses[-1], valAccuracy[-1]

            if step % saveInterval == 0:
                save(saveDir,
                     subjectName,
                     saver,
                     sess,
                     globalStep,
                     trainLosses,
                     trainAccuracy,
                     valLosses,
                     valAccuracy)

        testLosses = []
        testAccuracy = []

        for _ in xrange(testSteps):
            x, y = testDataGenerator.next()
            loss, acc = sess.run(layerFetches, feed_dict={
                layers['inputs']: x,
                layers['labels']: y,
                layers['is_training']: False,
                layers['keep_prob']: 1.0
            })

            testLosses.append(loss)
            testAccuracy.append(acc)

        print 'average testing', np.mean(testLosses), np.mean(testAccuracy)

        save(saveDir,
             subjectName,
             saver,
             sess,
             globalStep,
             trainLosses,
             trainAccuracy,
             valLosses,
             valAccuracy)

        graph.closeFig(fig1)
        graph.closeFig(fig2)


def save(saveDir, subjectName, saver, sess, globalStep, trainLosses, trainAccuracy, valLosses, valAccuracy):
    savePath = saveDir + '/{}'.format(subjectName)
    print 'saved to', saver.save(sess, savePath, global_step=globalStep)

    np.savez(savePath + '-graph-{}'.format(sess.run(globalStep)),
             trainLosses=trainLosses,
             trainAccuracy=trainAccuracy,
             valLosses=valLosses,
             valAccuracy=valAccuracy)


def load(loadDir, loadName, loadStep, saver, sess, globalStep):
    loadPath = loadDir + '/{}'.format(loadName)
    saver.restore(sess, loadPath + '-' + str(loadStep))
    print 'loaded ', loadPath + '-' + str(loadStep)

    data = np.load(loadPath + '-graph-{}.npz'.format(sess.run(globalStep)))

    trainLosses = data['trainLosses'].tolist()
    trainAccuracy = data['trainAccuracy'].tolist()
    valLosses = data['valLosses'].tolist()
    valAccuracy = data['valAccuracy'].tolist()

    return trainLosses, trainAccuracy, valLosses, valAccuracy


if __name__ == '__main__':
    pipeline_deepid()
