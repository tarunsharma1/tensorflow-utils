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
import cv2

from tensorflow.contrib.learn.python.learn.datasets import mnist

#############################################################################################################################

# Utils class to provide supporting operations. Has functions for
# 1. Reading data from tf records and return batches 
# 2. TODO : Add conv, relu, pool functionalities
# 3. Returns accuracy

###########################################################################################################################



class Utils:

	def read_and_decode(self,queue):
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(queue)
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
		image = tf.decode_raw(features['image_raw'],tf.float32)
		label = tf.decode_raw(features['label'], tf.float32)

		height = tf.cast(features['height'], tf.int32)
		width = tf.cast(features['width'], tf.int32)
		depth = tf.cast(features['depth'], tf.int32)

		image = tf.reshape(image, [height, width, depth])
		label.set_shape(10)
		return image,label



	def get_next_batch(self,batchsize,sess,img,label):
		# at some point replace this by tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
		batchx = []
		batchy = []
		for i in range(batchsize):
			img2,anno = sess.run([img,label])
			batchx.append(img2)
			batchy.append(anno)

		return batchx,batchy





	def get_accuracy(self,output,y):
		count = 0.0
		total = 0.0
		for i in range(0,len(output)):
			# in each batch
			pred = output[i]
			gt = y[i]
			total += len(pred)
			for k in range(0,len(pred)):
				if (np.argmax(pred[k]) == np.argmax(gt[k])):
					count+=1
				
		# total correct / (num_batches*images per batch)
		return count/(1.0*total)

			# correct_prediction = tf.equal(tf.argmax(gt, 1), tf.argmax(pred, 1))
	 	# 	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	 	# 	print(sess.run(accuracy, feed_dict={self.x: mnist.test.images, self.y: mnist.test.labels}))


