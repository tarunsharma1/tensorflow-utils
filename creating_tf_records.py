from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 

import sys

from tensorflow.contrib.learn.python.learn.datasets import mnist


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  # have to convert to float
  labels = np.array(labels,dtype=np.float32)
  #print (labels[0],type(labels[0]),labels[0].shape)
  num_examples = data_set.num_examples

  # 60000,28,28,1 and 60000,10
  # print (images.shape)
  # print (labels.shape)
  # sys.exit(0)

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = name + '.tfrecords'
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    labels_raw = labels[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        #'label': _int64_feature(int(labels[index])),
        'label': _bytes_feature(labels_raw),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


# Import data
mnist = mnist.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)



# data_sets = mnist.read_data_sets(FLAGS.directory,
#                                    dtype=tf.uint8,
#                                    reshape=False,
#                                    validation_size=FLAGS.validation_size)

# Convert to Examples and write the result to TFRecords.
convert_to(mnist.train, 'train')
#convert_to(data_sets.validation, 'validation')
#convert_to(data_sets.test, 'test')
