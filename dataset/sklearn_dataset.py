import cv2
from sklearn.datasets import lfw
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

dirname = os.path.dirname(__file__)

lfwData = {
    'lfwColor_0.5': lfw.fetch_lfw_people('{}/lfw/dataset'.format(dirname),
                                         min_faces_per_person=10,
                                         color=True),
    # 'lfwGray_0.5': lfw.fetch_lfw_people('{}/lfw/dataset'.format(dirname),
    #                                     min_faces_per_person=20,
    #                                     color=False)
}
# lfwData['lfwGray_0.5'].images = np.expand_dims(lfwData['lfwGray_0.5'].images, -1)

lfwDataGenerator = {
    'augmentedDataGenerator': ImageDataGenerator(featurewise_center=True,
                                               featurewise_std_normalization=True,
                                               data_format='channels_last',
                                               horizontal_flip=True,
                                               width_shift_range=0.1,
                                               height_shift_range=0.1,
                                               rotation_range=20,
                                               zoom_range=0.2),
    'defaultDataGenerator': ImageDataGenerator(featurewise_center=True,
                                               featurewise_std_normalization=True,
                                               data_format='channels_last',
                                               horizontal_flip=True)
}


def mapTargetToName(key, target):
    return lfwData[key].target_names[target]


def separateToTrainValTest(key, trainRatio, valRatio, testRatio):
    if os.path.exists('{}/lfw/{}.npy'.format(dirname, key)):
        indices = np.load('{}/lfw/{}.npy'.format(dirname, key))
        trainIndices = indices[0]
        valIndices = indices[1]
        testIndices = indices[2]

        trainX = lfwData[key].images[trainIndices]
        trainY = lfwData[key].target[trainIndices]

        valX = lfwData[key].images[valIndices]
        valY = lfwData[key].target[valIndices]

        testX = lfwData[key].images[testIndices]
        testY = lfwData[key].target[testIndices]

        return (trainX, trainY), (valX, valY), (testX, testY)

    if key not in lfwData:
        raise Exception('please create dataset for {}'.format(key))

    if sum([trainRatio, valRatio, testRatio]) != 1:
        raise Exception('sum of ratio parameter not equal 1')

    trainIndices = []
    valIndices = []
    testIndices = []

    for i, name in enumerate(lfwData[key].target_names):
        personIndices = np.argwhere(lfwData[key].target == i)
        np.random.seed(np.random.randint(100000))
        np.random.shuffle(personIndices)
        total = personIndices.shape[0]
        valLength = int(total * valRatio)
        testLength = int(total * testRatio)
        trainLength = total - valLength - testLength
        j = 0
        for _ in xrange(valLength):
            valIndices.append(personIndices[j])
            j += 1

        for _ in xrange(testLength):
            testIndices.append(personIndices[j])
            j += 1

        for _ in xrange(trainLength):
            trainIndices.append(personIndices[j])
            j += 1

    trainIndices = np.squeeze(np.vstack(trainIndices), axis=1)
    valIndices = np.squeeze(np.vstack(valIndices), axis=1)
    testIndices = np.squeeze(np.vstack(testIndices), axis=1)

    np.save('{}/lfw/{}'.format(dirname, key), [trainIndices, valIndices, testIndices])

    trainX, trainY = lfwData[key].images[trainIndices], lfwData[key].target[trainIndices]
    valX, valY = lfwData[key].images[valIndices], lfwData[key].target[valIndices]
    testX, testY = lfwData[key].images[testIndices], lfwData[key].target[testIndices]

    return (trainX, trainY), (valX, valY), (testX, testY)


def getClasses(key):
    return len(lfwData[key].target_names)


def getDataGenerator(dataGeneratorKey, X, Y, batchSize, shuffle, scaledSize=None, fit=False, trainKey=None):
    dataGenerator = lfwDataGenerator[dataGeneratorKey]

    # resize if needed
    if scaledSize:
        def resize(x):
            return cv2.resize(x, (scaledSize[1], scaledSize[0]))

        newX = np.zeros((X.shape[0], scaledSize[0], scaledSize[1], X.shape[3]), dtype=X.dtype)

        for i in xrange(X.shape[0]):
            newX[i] = resize(X[i])

        X = newX

    if fit:
        dataGenerator.fit(X)
        np.savez('{}/lfw/image_mean_and_std'.format(dirname),
                 mean=dataGenerator.mean,
                 std=dataGenerator.std) # for testing stage
    else:
        if trainKey:
            trainDataGenerator = lfwDataGenerator[trainKey]
            if trainDataGenerator.mean is not None:
                dataGenerator.mean = trainDataGenerator.mean
                dataGenerator.std = trainDataGenerator.std
            else:
                raise Exception('please create train data generator first')

    generator = dataGenerator.flow(X, Y,
                                   batchSize,
                                   shuffle,
                                   seed=np.random.randint(100000))
    return generator

