import numpy as np
import cv2
import os

import sys


def cropCenterFaceByRatio(image, widthRatio, heightRatio, croppedSize):
    height, width = image.shape[:2]
    leftRightRatio = widthRatio / 2
    topBottomRatio = heightRatio / 2
    centerX = width / 2
    centerY = height / 2
    leftRightWidth = width * leftRightRatio
    topBottomHeight = height * topBottomRatio
    x1, y1 = max(0, centerX - leftRightWidth), max(0, centerY - topBottomHeight)
    x2, y2 = min(centerX + leftRightWidth, width), min(centerY + topBottomHeight, height)
    return image[y1:y2, x1:x2]


def cropFaceByBoundingBoxFromDir(datadir, bboxes, croppedSize, saveDir):
    i = 1
    for root, dirs, files in os.walk(datadir):
        if files != []:
            for filename in files:
                imagePath = os.path.join(root, filename)
                classname = os.path.basename(os.path.dirname(imagePath))
                basename = os.path.basename(imagePath)
                basename, extension = os.path.splitext(basename)

                if extension.lower() == '.gif':
                    extension = '.png'

                if imagePath not in bboxes:
                    continue

                savepath = '{}/{}/{}{}'.format(saveDir, classname, basename, extension)
                sys.stdout.write('\r{} {}/{} image(s)'.format(savepath, i, len(bboxes)))
                sys.stdout.flush()
                i += 1

                if os.path.exists(savepath):
                    continue

                bbox = bboxes[imagePath]
                x1, y1, x2, y2 = bbox
                image = cv2.imread(imagePath)

                if image is None:
                    continue

                croppedImage = image[y1:y2, x1:x2]
                resizedImage = cv2.resize(croppedImage, croppedSize, interpolation=cv2.INTER_AREA)

                if not os.path.exists(os.path.dirname(savepath)):
                    os.makedirs(os.path.dirname(savepath))

                cv2.imwrite(savepath, resizedImage)
    print('done!')


def main():
    dirname = os.path.dirname(__file__)
    dataDir = '{}/facescrub'.format(dirname)
    saveDir = '{}/facescrub_crop'.format(dirname)
    bboxesPath = '{}/facescrub_bbox.txt'.format(dirname)
    bboxes = {}

    with open(bboxesPath) as f:
        for line in f:
            filepath, bbox = line.split('\t')
            bbox = [int(v) for v in bbox.split(',')]
            bboxes[filepath] = bbox

    cropFaceByBoundingBoxFromDir(dataDir, bboxes, (47, 55), saveDir)


if __name__ == '__main__':
    main()
