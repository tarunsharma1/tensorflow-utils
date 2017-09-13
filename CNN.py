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

from utils import Utils


class FNN:

	def mnist_init(self):
			
			self.x = tf.placeholder(tf.float32,[None,28,28,1])
			self.y = tf.placeholder(tf.float32,[None,10])
			self.TRAIN_FILE = '/home/tarun/mine/tensorflow_examples/tensorflow-utils/train.tfrecords'
			#self.TRAIN_FILE = '/home/tarun/tensorflow-utils/train.tfrecords'
			self.TEST_FILE = '/home/tarun/mine/tensorflow_examples/tensorflow-utils/train.tfrecords'
			self.batchsize = 128
			self.num_epochs = 10
			self.num_images = 60000
			self.num_test_images = 10000
			self.Utils = Utils()
			self.weights = {
			    # 5x5 conv, 1 input, 32 outputs
			    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
			    # 5x5 conv, 32 inputs, 64 outputs
			    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
			    # fully connected, 7*7*64 inputs, 1024 outputs
			    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
			    # 1024 inputs, 10 outputs (class prediction)
			    'out': tf.Variable(tf.random_normal([1024, 10]))
			}

			self.biases = {
			    'bc1': tf.Variable(tf.random_normal([32])),
			    'bc2': tf.Variable(tf.random_normal([64])),
			    'bd1': tf.Variable(tf.random_normal([1024])),
			    'out': tf.Variable(tf.random_normal([10]))
			}


	def conv2d(self,x, W, b, strides=1):
	    # Conv2D wrapper, with bias and relu activation
	    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	    x = tf.nn.bias_add(x, b)
	    return tf.nn.relu(x)


	def maxpool2d(self,x, k=2):
	    # MaxPool2D wrapper
	    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
	                          padding='SAME')		



	def model(self):
			# define model and return loss


			conv1 = self.conv2d(self.x, self.weights['wc1'], self.biases['bc1'])
			conv1 = self.maxpool2d(conv1, k=2)
		    # Convolution Layer
			conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
		    # Max Pooling (down-sampling)
			conv2 = self.maxpool2d(conv2, k=2)

		    # Fully connected layer
		    # Reshape conv2 output to fit fully connected layer input
			fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
			fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
			fc1 = tf.nn.relu(fc1)
		    # Apply Dropout
		    #fc1 = tf.nn.dropout(fc1, dropout)

		    # Output, class prediction
			output = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
			# output = tf.nn.softmax(tf.matmul(tf.reshape(self.x,[-1,784]),self.W)+self.b)
			loss = -tf.reduce_sum(self.y*tf.log(output))
			return loss,output


	def run(self):
			loss,_ = self.model()
			step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

			filename_queue = tf.train.string_input_producer([self.TRAIN_FILE])
			# variables for reading..used in get_next_batch()
			self.img, self.label = self.Utils.read_and_decode(filename_queue)

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
					batchx,batchy = self.Utils.get_next_batch(self.batchsize,sess,self.img,self.label)
					sess.run(step,{self.x:batchx,self.y:batchy})
				
				if (offset>0):	
					batchx,batchy = self.Utils.get_next_batch(offset,sess,self.img,self.label)
					sess.run(step,{self.x:batchx,self.y:batchy})
			return sess
			coord.request_stop()


	def test(self,sess):
		filename_queue = tf.train.string_input_producer([self.TEST_FILE])
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord,sess=sess)
		# variables for reading..used in get_next_batch()
		self.img, self.label = self.Utils.read_and_decode(filename_queue)
		loss,output = self.model()
		offset = int(self.num_test_images%self.batchsize)   
		num_loops = int(self.num_test_images/self.batchsize)
		
		output_vecs = []
		correct_labels = []

		# read test images
		for k in range(num_loops):
			batchx,batchy = self.Utils.get_next_batch(self.batchsize,sess,self.img,self.label)
			l,o = sess.run([loss,output],{self.x:batchx,self.y:batchy})
			output_vecs.append(o)
			correct_labels.append(batchy)
		
		if (offset>0):	
			batchx,batchy = self.Utils.get_next_batch(offset,sess,self.img,self.label)
			l,o = sess.run([loss,output],{self.x:batchx,self.y:batchy})
			output_vecs.append(o)
			correct_labels.append(batchy)
		
		#print (len(output_vecs),len(correct_labels))
		accuracy = self.Utils.get_accuracy(output_vecs,correct_labels)		
		print (accuracy)



instance = FNN() 
instance.mnist_init()
sess = instance.run()
instance.test(sess)
