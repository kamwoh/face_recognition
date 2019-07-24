import os

import numpy as np
import tensorflow as tf

import graph
from dataset import sklearn_dataset as sk
from dataset import facescrub_dataset as fs
from model import deep_id
import time


def pipeline_deepid():
    subjectName = 'face-recog-{}'.format(time.time())
    # subjectName = 'face-recog-{}'.format('1511344323.48')
    dirname = os.path.dirname(__file__)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    lr = 0.01
    # lr = 0.001
    # lr = 0.0001
    # lr = 0.00001
    batchSize = 500
    trainEpoch = 100

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

        print('train -', nTrainImages)
        print('val -', nValImages)
        print('test -', nTestImages)
        print('classes -', nClasses)

        trainSteps = trainEpoch * int(nTrainImages / batchSize)
        testSteps = int(nTestImages / batchSize)
        valSteps = int(nValImages / batchSize)

    fig1, ax1 = graph.getFigAx()
    fig2, ax2 = graph.getFigAx()

    with tf.Session(config=config) as sess:
        globalStep = tf.Variable(0, trainable=False)
        # learningRate = tf.train.exponential_decay(lr, globalStep, 100000, 0.5)
        learningRate = tf.constant(lr)

        valInterval = 1
        saveInterval = 500
        updateInterval = 500
        saveDir = '{}/trained/{}'.format(dirname, subjectName)

        layers, train_ops = deep_id.get_model(learningRate, nClasses, globalStep)

        # for node in tf.get_default_graph().as_graph_def().node:
        #     print node.name

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=None)

        layerFetches = [layers['loss'],
                        layers['accuracy']]
        fetches = layerFetches + train_ops.values()

        trainLosses = []
        trainAccuracy = []
        valLosses = []
        valAccuracy = []
        # trainLosses, trainAccuracy, valLosses, valAccuracy = load('{}/trained/{}'.format(dirname, subjectName),
        #                                                           subjectName,
        #                                                           49280,
        #                                                           saver,
        #                                                           sess,
        #                                                           globalStep)

        print('current learning rate', sess.run(learningRate))

        for step in range(trainSteps):
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

                print('---', step, '---')
                print('last train', trainLosses[-1], trainAccuracy[-1])
                print('last validation', valLosses[-1], valAccuracy[-1])

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

        testDataRes = []

        for tsteps in range(testSteps):
            x, y = testDataGenerator.next()

            loss, acc = sess.run(layerFetches, feed_dict={
                layers['inputs']: x,
                layers['labels']: y,
                layers['is_training']: False,
                layers['keep_prob']: 1.0
            })

            pred, softmax = sess.run([layers['ident_pred'], layers['ident_softmax']], feed_dict={
                layers['inputs']: x,
                layers['is_training']: False,
                layers['keep_prob']: 1.0
            })

            testDataRes.append((x, y, pred, softmax))

            testLosses.append(loss)
            testAccuracy.append(acc)

        print('average testing', np.mean(testLosses), np.mean(testAccuracy))

        # if not trainLFW:
        #     meanstdDir = '{}/dataset/facescrub_meanstd'.format(dirname)
        #     mean = np.load(meanstdDir + '/mean.npy')
        #     std = np.load(meanstdDir + '/std.npy')
        #
        #     saveErrorImages(testDataRes, mean, std)

        # save(saveDir,
        #      subjectName,
        #      saver,
        #      sess,
        #      globalStep,
        #      trainLosses,
        #      trainAccuracy,
        #      valLosses,
        #      valAccuracy)

        saveWeights(sess, '{}/trained_weight/{}'.format(dirname, subjectName))

        graph.closeFig(fig1)
        graph.closeFig(fig2)


def saveWeights(sess, weightsDir):
    weights = {}

    for v in tf.global_variables():
        print(v.name)
        weights[v.name] = sess.run(v)

    np.savez(weightsDir, **weights)
    print('saved to', weightsDir + '.npz')


def save(saveDir, subjectName, saver, sess, globalStep, trainLosses, trainAccuracy, valLosses, valAccuracy):
    savePath = saveDir + '/{}'.format(subjectName)
    print('saved to', saver.save(sess, savePath, global_step=globalStep))

    np.savez(savePath + '-graph-{}'.format(sess.run(globalStep)),
             trainLosses=trainLosses,
             trainAccuracy=trainAccuracy,
             valLosses=valLosses,
             valAccuracy=valAccuracy)


def load(loadDir, loadName, loadStep, saver, sess, globalStep):
    loadPath = loadDir + '/{}'.format(loadName)
    saver.restore(sess, loadPath + '-' + str(loadStep))
    print('loaded ', loadPath + '-' + str(loadStep))

    data = np.load(loadPath + '-graph-{}.npz'.format(sess.run(globalStep)))

    trainLosses = data['trainLosses'].tolist()
    trainAccuracy = data['trainAccuracy'].tolist()
    valLosses = data['valLosses'].tolist()
    valAccuracy = data['valAccuracy'].tolist()

    return trainLosses, trainAccuracy, valLosses, valAccuracy


def saveErrorImages(testDataRes, mean, std):
    import cv2
    dirname = os.path.dirname(__file__)
    i = 0
    for x, y, pred, softmax in testDataRes:
        x *= (std + 1e-7)
        x += mean
        x = x.astype(np.uint8)

        for j in range(x.shape[0]):
            image = cv2.cvtColor(x[j], cv2.COLOR_RGB2BGR)
            label = np.squeeze(y[j])
            p = np.squeeze(pred[j])
            s = np.squeeze(np.max(softmax[j]))
            path = '{}/result/{}/{}_{}_{}.png'.format(dirname, label, p, s, i)
            if np.equal(label, p):
                continue

            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            # print image
            cv2.imwrite(path, image)

            i += 1


if __name__ == '__main__':
    pipeline_deepid()
