import os
from sklearn.utils import shuffle
import numpy as np
import pickle


def load_train(train_path, image_size, num_channels, classes):
    images = []
    labels = []

    print('Going to read training images')
    train_file = os.path.join(train_path, 'train.pkl')
    raw_images, raw_labels = pickle.load(open(train_file, 'rb'))
    for i, img in enumerate(raw_images):
      index = raw_labels[i]
      image = img.reshape((image_size, image_size,num_channels))
      images.append(image)
      label = np.zeros(len(classes))
      label[index] = 1.0
      labels.append(label)
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


class DataSet(object):

  def __init__(self, images, labels):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end]


def read_train_sets(train_path, image_size, num_channels, classes, validation_size, test_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels = load_train(train_path, image_size, num_channels, classes)
  images, labels = shuffle(images, labels)

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  if isinstance(test_size, float):
    test_size = int(test_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]

  test_images = images[validation_size:(validation_size+test_size)]
  test_labels = labels[validation_size:(validation_size+test_size)]

  train_images = images[(validation_size+test_size):]
  train_labels = labels[(validation_size+test_size):]

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.valid = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)

  return data_sets


