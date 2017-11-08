import tensorflow as tf
import os
from tensorflow.contrib import slim
from sklearn.model_selection import KFold


def readPeopleAtLeastOne(filename):
    people = {}
    with open(filename) as f:
        n_people = int(f.readline()[:-1])
        i = 0
        for _ in xrange(n_people):
            name, n_images = f.readline()[:-1].split()
            people[name] = {'ind': i, 'n_images': n_images}
            i += 1

    return people


def readAll(train_people, test_people):
    alls = []
    dirname = os.path.dirname(__file__)
    for name, v in train_people.iteritems():
        ind = v['ind']
        n_images = int(v['n_images'])
        for i in xrange(1, n_images + 1):
            filename = '{}/lfw/{}/{}_{}.jpg'.format(dirname, name, name, str(i).zfill(4))
            rand = np.random.randint(1, 4)
            for _ in xrange(rand):
                alls.append({'filename': filename, 'label': ind})

    for name, v in test_people.iteritems():
        ind = v['ind']
        n_images = int(v['n_images'])
        for i in xrange(1, n_images + 1):
            filename = '{}/lfw/{}/{}_{}.jpg'.format(dirname, name, name, str(i).zfill(4))
            rand = np.random.randint(1, 6)
            for _ in xrange(rand):
                alls.append({'filename': filename, 'label': ind})

    return np.array(alls), len(train_people)+len(test_people)


def readPeople(filename):
    people = {}
    with open(filename) as f:
        n_people = int(f.readline()[:-1])
        for i in xrange(n_people):
            name, n_images = f.readline()[:-1].split()
            people[name] = i

    return people


def readPairs(filename, people):
    pairs = []
    dirname = os.path.dirname(__file__)
    with open(filename) as f:
        n_pairs = int(f.readline()[:-1])
        for _ in xrange(n_pairs):
            line = f.readline()[:-1]
            name, l, r = line.split()
            ind = people[name]
            l_filename = '{}/lfw/{}/{}_{}.jpg'.format(dirname, name, name, str(l).zfill(4))
            r_filename = '{}/lfw/{}/{}_{}.jpg'.format(dirname, name, name, str(r).zfill(4))

            pairs.append({
                'verif_label': 1,
                'l_label': ind,
                'r_label': ind,
                'l_filename': l_filename,
                'r_filename': r_filename
            })

        for _ in xrange(n_pairs):
            line = f.readline()[:-1]
            l_name, l, r_name, r = line.split()
            l_ind = people[l_name]
            r_ind = people[r_name]
            l_filename = '{}/lfw/{}/{}_{}.jpg'.format(dirname, l_name, l_name, str(l).zfill(4))
            r_filename = '{}/lfw/{}/{}_{}.jpg'.format(dirname, r_name, r_name, str(r).zfill(4))

            pairs.append({
                'verif_label': 0,
                'l_label': l_ind,
                'r_label': r_ind,
                'l_filename': l_filename,
                'r_filename': r_filename
            })

    return np.array(pairs)


import numpy as np

dirname = os.path.dirname(__file__)
# PEOPLE_TRAIN = readPeople('{}/label/peopleDevTrain.txt'.format(dirname))
# PEOPLE_TEST = readPeople('{}/label/peopleDevTest.txt'.format(dirname))
# PAIRS_TRAIN = readPairs('{}/label/pairsDevTrain.txt'.format(dirname), PEOPLE_TRAIN)
# PAIRS_TEST = readPairs('{}/label/pairsDevTest.txt'.format(dirname), PEOPLE_TEST)
PEOPLE_TRAIN = readPeopleAtLeastOne('{}/label/peopleDevTrain.txt'.format(dirname))
PEOPLE_TEST = readPeopleAtLeastOne('{}/label/peopleDevTest.txt'.format(dirname))
ALL, TOTAL_PEOPLE = readAll(PEOPLE_TRAIN, PEOPLE_TEST)

kf = KFold(n_splits=5, shuffle=True)
ALL_TRAIN = []
ALL_TEST = []

for train_index, test_index in kf.split(ALL):
    ALL_TRAIN.append(ALL[train_index])
    ALL_TEST.append(ALL[test_index])
