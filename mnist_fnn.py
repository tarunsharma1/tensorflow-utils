from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import skimage.io as io

import tensorflow as tf

import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data as mnist_data

#mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)



def read_and_decode(queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.string),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64)
      })
	#image = np.fromstring(features['image_raw'], dtype=np.uint8)
	image = tf.decode_raw(features['image_raw'],tf.float32)
	label = tf.decode_raw(features['label'], tf.float32)

	height = tf.cast(features['height'], tf.int32)
	width = tf.cast(features['width'], tf.int32)
	depth = tf.cast(features['depth'], tf.int32)

	#image_shape = tf.pack([height, width, depth])
	image2 = tf.reshape(image, [height, width, depth])
	return image2,label





TRAIN_FILE = '/home/tarun/mine/tensorflow_examples/tensorflow-utils/train.tfrecords'


num_epochs = 1
filename_queue = tf.train.string_input_producer(
        [TRAIN_FILE])

    # Even when reading in multiple threads, share the filename
    # queue.

img,label = read_and_decode(filename_queue)


b = tf.Variable(tf.zeros([10]))
W = tf.Variable(tf.zeros([784,10]))

x = tf.placeholder(tf.float32,[None,28,28,1])
y = tf.placeholder(tf.float32,[None,10])


output = tf.nn.softmax(tf.matmul(tf.reshape(x,[-1,784]),W)+b)

loss = -tf.reduce_sum(y*tf.log(output)) 

#step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)

img,anno = sess.run([img,label])


print (img.shape)

#print(img[0, :, :, :].shape)
#print (sess.run(label))

# Wait for threads to finish.
#coord.join(threads)
#sess.close()


sys.exit(0)













for i in range(1000):
	#batchx, batchy = mnist.train.next_batch(100)
	reader = tf.TFRecordReader()

	_, serialized_example = reader.read(TRAIN_FILE)
	
	import sys
	sys.exit(0)
	
	features = tf.parse_single_example(
	  serialized_example,
	  # Defaults are not specified since both keys are required.
	  features={
	      'image_raw': tf.FixedLenFeature([], tf.string),
	      'label': tf.FixedLenFeature([], tf.int64),
	  })

	# Convert from a scalar string tensor (whose single string has
	# length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
	# [mnist.IMAGE_PIXELS].
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	image.set_shape([mnist.IMAGE_PIXELS])

	# OPTIONAL: Could reshape into a 28x28 image and apply distortions
	# here.  Since we are not applying any distortions in this
	# example, and the next step expects the image to be flattened
	# into a vector, we don't bother.

	# Convert from [0, 255] -> [-0.5, 0.5] floats.
	image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

	# Convert label from a scalar uint8 tensor to an int32 scalar.
	label = tf.cast(features['label'], tf.int32)



	sess.run(step,{x:batchx,y:batchy})
#	print sess.run(loss,{x:batchx,y:batchy})


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels}))
testx, testy = mnist.test.next_batch(1)
print (sess.run(W))
#print(sess.run(output,feed_dict={x:testx,y:testy}))
