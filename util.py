# coding: utf-8

import warnings
import os
import csv

import numpy as np
import cv2 as cv

from itertools import chain
from sklearn.model_selection import _split
from scipy.misc import imsave, imread, imresize

__author__ = "Dariusz Brzezinski"


class ForcedStratifiedShuffleSplit(_split.BaseShuffleSplit):
    def __init__(self, n_splits=10, test_size="default", train_size=None,
                 random_state=None):
        super(ForcedStratifiedShuffleSplit, self).__init__(
            n_splits, test_size, train_size, random_state)

    def _iter_indices(self, X, y, groups=None):
        n_samples = _split._num_samples(X)
        y = _split.check_array(y, ensure_2d=False, dtype=None)
        n_train, n_test = _split._validate_shuffle_split(n_samples, self.test_size,
                                                         self.train_size)

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([' '.join(row.astype('str')) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(np.argsort(y_indices, kind='mergesort'),
                                 np.cumsum(class_counts)[:-1])

        rng = _split.check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _split._approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _split._approximate_mode(class_counts_remaining, n_test, rng)

            train = []
            test = []

            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation,
                                                             mode='clip')

                train.extend(perm_indices_class_i[:n_i[i]])
                test.extend(perm_indices_class_i[n_i[i]:n_i[i] + t_i[i]])

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test


def train_test_split(*arrays, **options):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    test_size = options.pop('test_size', 'default')
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    stratify = options.pop('stratify', None)
    shuffle = options.pop('shuffle', True)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    if test_size == 'default':
        test_size = None
        if train_size is not None:
            warnings.warn("From version 0.21, test_size will always "
                          "complement train_size unless both "
                          "are specified.",
                          FutureWarning)

    if test_size is None and train_size is None:
        test_size = 0.25

    arrays = _split.indexable(*arrays)

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for "
                "shuffle=False")

        n_samples = _split._num_samples(arrays[0])
        n_train, n_test = _split._validate_shuffle_split(n_samples, test_size,
                                                  train_size)

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        if stratify is not None:
            CVClass = ForcedStratifiedShuffleSplit
        else:
            CVClass = _split.ShuffleSplit

        cv = CVClass(test_size=test_size,
                     train_size=train_size,
                     random_state=random_state)

        train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(chain.from_iterable((_split.safe_indexing(a, train),
                                     _split.safe_indexing(a, test)) for a in arrays))


def resize_images(image_files, width, height):
    for image_file in image_files:
        img = imread(image_file)
        img = imresize(img, (height, width))
        imsave(image_file[:-3] + str(width) + "x" + str(height) + ".png", img)


def hsv2rgb(h, s, v):
    c = s*v
    x = c * (1 - abs(h/60 % 2 - 1))
    tab = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)]
    ans = tab[int(h/60) % 6]
    ans = [255 * float(i+v-c) for i in ans]
    return ans


def linear_interpolation(val, lb, ub, nlb, nub):
    return nlb + (val-lb)/(ub-lb) * (nub-nlb)


def show_img(image, window_name='image', width=600, height=600):
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, width, height)
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def normalize_gray_image(img, base=8):
    base = pow(2, base) - 1
    float_array = np.array(img, dtype=np.float64)
    float_array -= float_array.min()
    float_array *= float(base) / float_array.max()
    return np.array(np.around(float_array), dtype=np.uint8)


def negative(img):
    return 255 - img


def closest_pixel(coord):
    fx, fy = coord
    return int(round(fx)), int(round(fy))


def bresenham_line_points(x1, y1, x2, y2):
    """
    Zwraca listę punktów, przez które przechodzić będzie prosta
    o zadanym początku i końcu

    Parametry
    ----------
    x1, y1, x2, y2 : int
        (x1, y1) - punkt poczatkowy
        (x2, y2) - punkt końcowy

    """
    # Zmienne pomocnicze
    d = dx = dy = ai = bi = xi = yi = 0
    x = x1
    y = y1
    points = []

    # Ustalenie kierunku rysowania
    xi = 1 if x < x2 else -1
    dx = abs(x1 - x2)

    # Ustalenie kierunku rysowania
    yi = 1 if y1 < y2 else -1
    dy = abs(y1 - y2)

    # Pierwszy piksel
    points.append((x, y))

    ai = -1 * abs(dy - dx) * 2
    bi = min(dx, dy) * 2
    d = bi - max(dx, dy)

    # Oś wiodąca OX
    if dx > dy:
        while x != x2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                x += xi

            points.append((x, y))

    # Oś wiodąca OY
    else:
        while y != y2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                y += yi

            points.append((x, y))

    return points


def center_of_mass(img):
    if True:    #FIXME remove
        row_centers = []
        for row in img:
            s0 = sum(row)
            s1 = 0
            for i, pixel in enumerate(row):
                s1 += i
                if s1 > s0:
                    row_centers.append([i, s0])
                    break
        s0 = sum([i[1] for i in row_centers])
        s1 = 0
        for y, (x, w) in enumerate(row_centers):
            s1 += w
            if s1 > s0:
                #FIXME Point notation?
                return y, x

    # TODO
    if cv.moments(img)['m00'] != 0:
        m = cv.moments(img)
        x = m['m10'] / m['m00']
        y = m['m01'] / m['m00']
    else:
        return img.shape[0]/2, img.shape[1]/2
    return round(x), round(y) # OK


def write_to_csv(filename, values):
    flag = 'w' if not os.path.isfile(filename) else 'a'
    with open(filename, flag) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(values.keys()))
        if flag == 'w':
            writer.writeheader()
        writer.writerow(values)

