
import os
import numpy as np
import shutil

import sys


def isValidExtension(path):
    extension = os.path.splitext(path)[1]
    return extension.lower() in ['.jpg', '.jpeg', '.png', '.gif']


def separateToTrainValTest(datadir, trainRatio, valRatio, testRatio):
    if sum([trainRatio, valRatio, testRatio]) != 1:
        raise Exception('sum of ratio parameter not equal 1')


    for root, dirs, files in os.walk(datadir):
        if dirs != []:
            for dir in dirs:
                dirpath = os.path.join(root, dir)
                filenames = os.listdir(dirpath)

                np.random.seed(np.random.randint(100000))
                np.random.shuffle(filenames)

                total = len(filenames)
                valLength = int(total * valRatio)
                testLength = int(total * testRatio)
                trainLength = total - valLength - testLength

                sys.stdout.write('\rprocessing {}'.format(dir))
                sys.stdout.flush()

                j = 0

                if not os.path.exists(os.path.join(datadir + '_val', dir)):
                    os.makedirs(os.path.join(datadir + '_val', dir))

                if not os.path.exists(os.path.join(datadir + '_test', dir)):
                    os.makedirs(os.path.join(datadir + '_test', dir))

                if not os.path.exists(os.path.join(datadir + '_train', dir)):
                    os.makedirs(os.path.join(datadir + '_train', dir))

                for _ in range(valLength):
                    filepath = os.path.join(root, dir, filenames[j])
                    newfilepath = os.path.join(datadir + '_val', dir, filenames[j])
                    if not os.path.exists(newfilepath):
                        shutil.copy(filepath, newfilepath)
                    j += 1

                for _ in range(testLength):
                    filepath = os.path.join(root, dir, filenames[j])
                    newfilepath = os.path.join(datadir + '_test', dir, filenames[j])
                    if not os.path.exists(newfilepath):
                        shutil.copy(filepath, newfilepath)
                    j += 1

                for _ in range(trainLength):
                    filepath = os.path.join(root, dir, filenames[j])
                    newfilepath = os.path.join(datadir + '_train', dir, filenames[j])
                    if not os.path.exists(newfilepath):
                        shutil.copy(filepath, newfilepath)
                    j += 1

    print('done')

def main():
    dirname = os.path.dirname(__file__)
    dataDir = '{}/facescrub_crop'.format(dirname)
    separateToTrainValTest(dataDir, 0.7, 0.1, 0.2)

if __name__ == '__main__':
    main()