import numpy as np
import os
import cv2

from keras.preprocessing.image import ImageDataGenerator

dirname = os.path.dirname(__file__)
trainDatasetPath = '{}/facescrub_train'.format(dirname)
testDatasetPath = '{}/facescrub_test'.format(dirname)
valDatasetPath = '{}/facescrub_val'.format(dirname)


def getDataGenerator(datadir, meanstddir, batchSize, shuffle):
    dataGenerator = ImageDataGenerator(featurewise_center=True,
                                       featurewise_std_normalization=True,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.1,
                                       rotation_range=10,
                                       horizontal_flip=True)

    mean = np.load(meanstddir + '/mean.npy')
    std = np.load(meanstddir + '/std.npy')

    nImages = 0
    classes = []
    for subdir in sorted(os.listdir(datadir)):
        if os.path.isdir(os.path.join(datadir, subdir)):
            classes.append(subdir)
            nImages += len(os.listdir(os.path.join(datadir, subdir)))

    # need to swap channel
    mean[0, 0, :] = mean[0, 0, ::-1]
    std[0, 0, :] = mean[0, 0, ::-1]

    dataGenerator.mean = mean
    dataGenerator.std = std

    generator = dataGenerator.flow_from_directory(datadir,
                                                  target_size=(55, 47),
                                                  batch_size=batchSize,
                                                  shuffle=shuffle,
                                                  class_mode='sparse',
                                                  classes=classes,
                                                  seed=np.random.randint(100000))
    return generator, nImages
