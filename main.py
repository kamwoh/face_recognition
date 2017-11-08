from dataset import data_generator as data
from model import deep_id
import tensorflow as tf
import numpy as np
import cv2


def pipeline_deepid():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        layers, train_ops = deep_id.get_model(0.0005, data.TOTAL_PEOPLE)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        fetches = [layers['accuracy'],
                   layers['loss']] + train_ops.values()

        curr = 0
        batch_size = 32
        n_epoch = 20
        for e in xrange(1, n_epoch + 1):
            avg_acc = 0
            avg_loss = 0
            curr_fold = np.random.randint(len(data.ALL_TRAIN))
            curr_train = data.ALL_TRAIN[curr_fold]
            n_steps = int(len(curr_train) / batch_size)
            for step in xrange(n_steps):
                inputs, labels, curr = get_all_data(curr_train, curr, batch_size)
                acc, loss = sess.run(fetches, feed_dict={
                    layers['inputs']: inputs,
                    layers['labels']: labels,
                    layers['is_training']: True
                })[:2]

                avg_acc += acc
                avg_loss += loss

            avg_acc /= n_steps
            avg_loss /= n_steps
            print 'e: ', e
            print 'accuracy: ', avg_acc
            print 'loss: ', avg_loss

            if e % 5 == 0:
                curr = 0
                avg_acc = 0
                avg_loss = 0
                curr_test = data.ALL_TEST[curr_fold]
                n_steps = int(len(curr_test) / batch_size)
                for step in xrange(n_steps):
                    inputs, labels, curr = get_all_data(curr_test, curr, batch_size)
                    acc, loss, argmax = sess.run([layers['accuracy'],
                                                  layers['loss'],
                                                  layers['argmax']], feed_dict={
                        layers['inputs']: inputs,
                        layers['labels']: labels,
                        layers['is_training']: False
                    })[:3]

                    avg_acc += acc
                    avg_loss += loss

                avg_acc /= n_steps
                avg_loss /= n_steps
                print 'e: ', e
                print 'test accuracy: ', avg_acc
                print 'test loss: ', avg_loss
                saver.save(sess, 'trained/deepid2-{}'.format(e))

        print 'done'


def pipeline():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        layers, train_ops = deep_id.get_model(0.0005, len(data.PEOPLE_TRAIN))
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        fetches = [layers['verif_accuracy'],
                   layers['ident_loss'],
                   layers['verif_loss']] + train_ops.values()
        curr = 0
        batch_size = 32
        n_epoch = 100
        for e in xrange(1, n_epoch + 1):
            n_steps = int(len(data.PAIRS_TRAIN) / batch_size)
            avg_acc = 0
            avg_loss_1 = 0
            avg_loss_2 = 0
            for step in xrange(n_steps):
                l_inputs, \
                r_inputs, \
                l_labels, \
                r_labels, \
                verif_labels, \
                curr = get_data(data.PAIRS_TRAIN, curr, batch_size)
                acc, loss_1, loss_2 = sess.run(fetches, feed_dict={
                    layers['l_input']: l_inputs,
                    layers['r_input']: r_inputs,
                    layers['l_labels']: l_labels,
                    layers['r_labels']: r_labels,
                    layers['verif_labels']: verif_labels,
                })[:3]

                avg_acc += acc
                avg_loss_1 += loss_1
                avg_loss_2 += loss_2

            avg_acc /= n_steps
            avg_loss_1 /= n_steps
            avg_loss_2 /= n_steps
            print 'average train verification accuracy: ', avg_acc
            print 'ident: ', avg_loss_1
            print 'verif: ', avg_loss_2

            # if e % 5 == 0:
            #     curr = 0
            #     n_steps = int(len(data.PEOPLE_TEST) / batch_size)
            #     avg_acc = 0
            #     for step in xrange(n_steps):
            #         l_inputs, \
            #         r_inputs, \
            #         l_labels, \
            #         r_labels, \
            #         verif_labels, \
            #         curr = get_data(data.PAIRS_TRAIN, curr, batch_size)
            #         acc = sess.run(fetches, feed_dict={
            #             layers['l_input']: l_inputs,
            #             layers['r_input']: r_inputs,
            #             layers['l_labels']: l_labels,
            #             layers['r_labels']: r_labels,
            #             layers['verif_labels']: verif_labels,
            #         })[0]
            #
            #         avg_acc += acc
            #         print 'test verification accuracy: ', acc
            #
            #     avg_acc /= n_steps
            #     print 'average test verification accuracy: ', avg_acc
            #     saver.save(sess, 'trained/deepid2-{}'.format(e))

        print 'done'


def random_crop(in_data, size):
    h, w = in_data.shape[:2]

    rand_x = np.random.randint(0, w - size[1])
    rand_y = np.random.randint(0, h - size[0])

    return in_data[rand_y:rand_y + size[0], rand_x:rand_x + size[1]]


def random_scale(in_data, lower, upper, size):
    fx = np.random.uniform(lower, upper)
    fy = np.random.uniform(lower, upper)
    out_data = cv2.resize(in_data, None, fx=fx, fy=fy)
    return cv2.resize(out_data, size)


def get_all_data(DATA, curr, batch_size):
    n_total = len(DATA)

    if curr + batch_size >= n_total or curr == 0:
        shuffle_ind = np.arange(n_total)
        np.random.seed(np.random.randint(100000))
        np.random.shuffle(shuffle_ind)

        curr = 0
        DATA[:] = DATA[shuffle_ind]

    inputs = []
    labels = []
    datum = DATA[curr:curr + batch_size]

    for i in xrange(batch_size):
        d = datum[i]
        filename = d['filename']
        label = d['label']
        img = cv2.imread(filename)
        img = random_scale(img, 1, 1, (72, 85))
        img = random_crop(img, (55, 47))
        # img = (img - img.mean()) / img.std()
        inputs.append(img)
        labels.append(label)

    labels = np.vstack(labels)

    return inputs, labels, curr + batch_size


def get_data(DATA, curr, batch_size):
    n_total = len(data.PAIRS_TRAIN)

    if curr + batch_size >= n_total or curr == 0:
        shuffle_ind = np.arange(n_total)
        np.random.seed(np.random.randint(100000))
        np.random.shuffle(shuffle_ind)

        curr = 0
        DATA[:] = DATA[shuffle_ind]

    l_inputs = []
    r_inputs = []
    l_labels = []
    r_labels = []
    verif_labels = []
    pairs = DATA[curr:curr + batch_size]

    for i in xrange(batch_size):
        pair = pairs[i]
        l_filename = pair['l_filename']
        r_filename = pair['r_filename']
        l_label = pair['l_label']
        r_label = pair['r_label']
        verif_label = pair['verif_label']

        # preprocessing
        l_img = cv2.imread(l_filename)
        l_img = random_scale(l_img, 0.9, 1.1, (150, 150))
        l_img = random_crop(l_img, (55, 47))
        # l_img = cv2.resize(l_img, (47,55))
        l_img = (l_img - l_img.mean()) / l_img.std()
        l_inputs.append(l_img)

        r_img = cv2.imread(r_filename)
        r_img = random_scale(r_img, 0.9, 1.1, (150, 150))
        r_img = random_crop(r_img, (55, 47))
        # r_img = cv2.resize(r_img, (47, 55))
        r_img = (r_img - r_img.mean()) / r_img.std()
        r_inputs.append(r_img)

        l_labels.append(l_label)
        r_labels.append(r_label)

        verif_labels.append(verif_label)

    l_labels = np.vstack(l_labels)
    r_labels = np.vstack(r_labels)
    verif_labels = np.vstack(verif_labels)

    return l_inputs, r_inputs, l_labels, r_labels, verif_labels, curr + batch_size


if __name__ == '__main__':
    pipeline_deepid()
