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
			self.b = tf.Variable(tf.zeros([10]))
			self.W = tf.Variable(tf.zeros([784,10]))

			self.x = tf.placeholder(tf.float32,[None,28,28,1])
			self.y = tf.placeholder(tf.float32,[None,10])
			self.TRAIN_FILE = '/home/tarun/mine/tensorflow_examples/tensorflow-utils/train.tfrecords'
			self.batchsize = 100
			self.num_epochs = 10
			self.num_images = 10000
			self.num_test_images = 2000
			self.Utils = Utils()


	def model(self):
			# define model and return loss...this along with mnist_init will be moved to another script
			output = tf.nn.softmax(tf.matmul(tf.reshape(self.x,[-1,784]),self.W)+self.b)
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
					batchx,batchy = self.get_next_batch(offset,sess,self.img,self.label)
					sess.run(step,{self.x:batchx,self.y:batchy})
			return sess
			coord.request_stop()


	def test(self,sess):
		filename_queue = tf.train.string_input_producer([self.TRAIN_FILE])
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
			batchx,batchy = self.get_next_batch(offset,sess,self.img,self.label)
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
