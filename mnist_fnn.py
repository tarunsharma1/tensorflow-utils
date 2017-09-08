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

#############################################################################################################################

# Utils class to provide supporting operations. Has functions for
# 1. Reading data from tf records and return batches 
# 2. TODO : Add conv, relu, pool functionalities
# 3. TODO : Move all mnist varibles, equations, and even run function to other script


###########################################################################################################################



class Utils:
	def mnist_init(self):
		self.b = tf.Variable(tf.zeros([10]))
		self.W = tf.Variable(tf.zeros([784,10]))

		self.x = tf.placeholder(tf.float32,[None,28,28,1])
		self.y = tf.placeholder(tf.float32,[None,10])
		self.TRAIN_FILE = '/home/tarun/mine/tensorflow_examples/tensorflow-utils/train.tfrecords'
		self.batchsize = 5
		self.num_epochs = 5
		self.num_images = 10000


	def read_and_decode(self,queue):
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(self.filename_queue)
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
		label3 = tf.decode_raw(features['label'], tf.float32)

		height = tf.cast(features['height'], tf.int32)
		width = tf.cast(features['width'], tf.int32)
		depth = tf.cast(features['depth'], tf.int32)

		#image_shape = tf.pack([height, width, depth])
		image2 = tf.reshape(image, [height, width, depth])
		return image2,label3


	#img,label = read_and_decode(filename_queue)


	def model(self):
		# define model and return loss...this along with mnist_init will be moved to another script
		output = tf.nn.softmax(tf.matmul(tf.reshape(self.x,[-1,784]),self.W)+self.b)
		loss = -tf.reduce_sum(self.y*tf.log(output))
		return loss



	def get_next_batch(self,batchsize,sess):
		batchx = []
		batchy = []
		for i in range(batchsize):
			img,anno = sess.run([self.img,self.label2])
			print (anno)
			sys.exit(0)
			batchx.append(img)
			batchy.append(anno)

		return batchx,batchy



	def run(self):
		loss = self.model()
		step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

		self.filename_queue = tf.train.string_input_producer([self.TRAIN_FILE])
		# variables for reading..used in get_next_batch()
		self.img, self.label2 = self.read_and_decode(self.filename_queue)


		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)

		# for reading the tf record
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord,sess=sess)

		
		offset = int(self.num_images%self.batchsize)   
		num_loops = int(self.num_images/self.batchsize)

		
		for i in range(self.num_epochs):
			print (" ### in epoch " + str(i) + "###")
			for k in range(num_loops):
				batchx,batchy = self.get_next_batch(self.batchsize,sess)
				print (batchx[0][:,:,0].shape)
				#cv2.imshow('window',batchx[0][:,:,0])
				#cv2.waitKey(0)
				#print (len(batchx),batchx[0],batchy[0])
				sess.run(step,{self.x:batchx,self.y:batchy})
				sys.exit(0)
				
			batchx,batchy = self.get_next_batch(offset,sess)
			print (len(batchx),batchx[0].shape,batchy[0])
			sys.exit(0)
			sess.run(step,{x:batchx,y:batchy})
	


	# def get_accuracy():
	# 	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
	# 	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# 	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))





instance = Utils() 
instance.mnist_init()
instance.run()











	
#	print sess.run(loss,{x:batchx,y:batchy})



# testx, testy = mnist.test.next_batch(1)
# print (sess.run(W))
# #print(sess.run(output,feed_dict={x:testx,y:testy}))
