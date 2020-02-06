# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:13:42 2017

@author: MinaMelek
"""

import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle

valid_images = [".jpg",".jpeg",".png",".tga",".bmp"]
def load_train(train_path, height, width, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld)
        files = os.listdir(path)
        for fl in files:
            ext = os.path.splitext(fl)[1]
            if ext.lower() not in valid_images:
                continue
            image = cv2.imread(os.path.join(path,fl))
            image = image[40:-40,79:-79]
            image = cv2.resize(image,(width,height))
            image = image/255
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


def load_test(test_path, height, width,classes):
    X_test = []
    X_test_id = []  
#    for fld in classes:
#        index = classes.index(fld)
#        print('Loading {} files (Index: {})'.format(fld, index))
#        path = os.path.join(test_path,fld)
#        files = os.listdir(path)
#        print("Reading test images")
#        for fl in files:
###            print(fl)
#            ext = os.path.splitext(fl)[1]
#            if ext.lower() not in valid_images:
#                continue
#            flbase = os.path.basename(fl)
#            image = cv2.imread(os.path.join(path,fl))
#            image = image[40:-40,79:-79]
#            image = cv2.resize(image,(width,height))
#            image = image/255
#            X_test.append(image)
#            X_test_id.append(flbase)
#
#  ### because we're not creating a DataSet object for the test images, normalization happens here
#        X_test = np.array(X_test, dtype=np.float32)
#        X_test_id = np.array(X_test_id)
    return X_test, X_test_id



class DataSet(object):

  def __init__(self, images, labels, ids, cls):
    self._num_examples = images.shape[0]
    images = images.astype(np.float32)
#    images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._ids = ids
    self._cls = cls
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def ids(self):
    return self._ids

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # # Shuffle the data (maybe)
      # perm = np.arange(self._num_examples)
      # np.random.shuffle(perm)
      # self._images = self._images[perm]
      # self._labels = self._labels[perm]
      # Start next epoch

      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, height, width, classes, validation_size=0):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, ids, cls = load_train(train_path, height, width, classes)
  images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_ids = ids[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_ids = ids[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

  return data_sets


def read_test_set(test_path, height, width,classes):
  images, ids  = load_test(test_path, height, width,classes)
  return images, ids