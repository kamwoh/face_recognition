import os
import cv2
import numpy as np
import sys


def computeImageMean(datadir, grayscale=False):
    channel = 1 if grayscale else 3
    mean = np.zeros((1, 1, channel), dtype=np.float64)
    nImages = 0
    imageShape = None

    print 'computing image mean'

    for root, dirs, files in os.walk(datadir):
        if files != []:
            for filename in files:
                filepath = os.path.join(root, filename)

                sys.stdout.write('\rprocessing {}'.format(filepath))
                sys.stdout.flush()

                if grayscale:
                    image = cv2.imread(filepath, 0)
                else:
                    image = cv2.imread(filepath)

                imageShape = image.shape
                imageMean = np.mean(image, axis=(0, 1)).reshape((1, 1, channel))

                mean += imageMean
                nImages += 1

    mean /= nImages
    print
    return mean, nImages, imageShape


def computeImageStd(datadir, mean, nImages, imageShape, grayscale=False):
    channel = 1 if grayscale else 3
    sumsquare = np.zeros((imageShape[0], imageShape[1], channel), dtype=np.float64)
    rcount = nImages * imageShape[0] * imageShape[1]

    print 'computing image std'

    for root, dirs, files in os.walk(datadir):
        if files != []:
            for filename in files:
                filepath = os.path.join(root, filename)

                sys.stdout.write('\rprocessing {}'.format(filepath))
                sys.stdout.flush()

                if grayscale:
                    image = cv2.imread(filepath, 0)
                else:
                    image = cv2.imread(filepath)

                image_mean = image - mean  # image minus mean
                image_mean = np.square(image_mean)

                sumsquare += image_mean

    sumsquare = np.sum(sumsquare, axis=(0, 1), keepdims=True)
    variance = sumsquare / rcount
    std = np.sqrt(variance)
    print
    return std

def main():
    dirname = os.path.dirname(__file__)
    imageMeanPath = '{}/facescrub_meanstd/mean'.format(dirname)
    imageStdPath = '{}/facescrub_meanstd/std'.format(dirname)

    imageDir = '{}/facescrub_crop_train'.format(dirname)

    mean, nImages, imageShape = computeImageMean(imageDir)
    std = computeImageStd(imageDir, mean, nImages, imageShape)

    if not os.path.exists(os.path.dirname(imageMeanPath)):
        os.makedirs(os.path.dirname(imageMeanPath))

    if not os.path.exists(os.path.dirname(imageStdPath)):
        os.makedirs(os.path.dirname(imageStdPath))

    np.save(imageMeanPath, mean)
    np.save(imageStdPath, std)

if __name__ == '__main__':
    main()