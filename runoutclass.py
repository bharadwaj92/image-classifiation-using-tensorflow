from scipy.misc import imread
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib as plt
import numpy as np
import os
from sys import stdin
from PIL import Image
import time
import datetime as dt
notpath = r'/mnt/c/datasets/tensorflow/imageclass/nonrunouts'
outpath = r'/mnt/c/datasets/tensorflow/imageclass/runouts'
labels = []
num_examples = 0
img_size = 500
num_channels = 1
img_pix =  img_size * img_size * num_channels
img_shape = (img_size, img_size)
maxheight = 0
maxwidth = 0
num_classes = 2 
train_images = []
train_labels = []
print("enter batch size for training")
batch_size = int(stdin.readline().strip())
total_iterations = 0
epochs_completed = 0 
index_in_epoch = 0
print("enter the number of iterations your want to perform")
num_iterations = int(stdin.readline().strip())
#Network configuration
# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 49         # There are 36 of these filters.

#Convolution Layer 3.

filter_size3 = 5
num_filters3 = 64

# Fully-connected layer.
fc_size = 16  

def create_traindata():
	global train_images
	global train_labels
	global num_examples
	for filename in os.listdir(notpath):
		image = Image.open(os.path.join(notpath,filename))
		image = image.resize((img_size,img_size))
		train_images.append(np.array(image))
		labels.append(0)
	for filename in os.listdir(outpath):
		image = Image.open(os.path.join(outpath,filename))
		image = image.resize((img_size,img_size))
		train_images.append(np.array(image))
		labels.append(1)
	train_images = np.array(train_images)
	#print(np.shape(train_images))
	train_images = train_images.reshape(len(labels), img_pix)
	train_labels = np.array(labels)
	num_examples = len(labels)

def create_placeholders(batch_size):
	x = tf.placeholder(tf.float32 , shape = [None,img_pix] , name = "x")
	ximages = tf.reshape(x,[-1,img_size, img_size,num_channels])
	ylabels = tf.placeholder(tf.float32 , shape = [None])
	ylabels_ce = tf.placeholder(tf.int32, shape = [None])
	return x, ximages , ylabels , ylabels_ce

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))
	
def new_conv_layer(inputx, num_input_channels,filter_size,num_filters,use_pooling = True):
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	weights = new_weights(shape=shape)
	biases = new_biases(length=num_filters)
	layer = tf.nn.conv2d(input = inputx , filter = weights,strides = [1,1,1,1],padding = 'SAME')
	layer += biases
	if(use_pooling):
		layer = tf.nn.max_pool(value = layer , ksize = [1,5,5,1],strides = [1,5,5,1],padding = 'SAME')
	layer = tf.nn.relu(layer)
	return layer,weights

def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat, num_features

def new_fc_layer(input , num_inputs , num_outputs , use_relu = True):
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)
	layer = tf.matmul(input, weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)
	return layer

def compu_graph(x_batch):
	layer_conv1, weights_conv1 = new_conv_layer(inputx = x_batch,num_input_channels = num_channels, filter_size=filter_size1 , num_filters = num_filters1, use_pooling = True)
	layer_conv2, weights_conv2 = new_conv_layer(inputx = layer_conv1,num_input_channels = num_filters1, filter_size=filter_size2 , num_filters = num_filters2, use_pooling = True)
	layer_conv3, weights_conv3 = new_conv_layer(inputx = layer_conv2,num_input_channels = num_filters2, filter_size=filter_size3 , num_filters = num_filters3, use_pooling = True)
	layer_flat, num_features = flatten_layer(layer_conv3)
	layer_fc1 = new_fc_layer(layer_flat ,num_inputs = num_features, num_outputs = fc_size, use_relu = True)
	layer_fc2 = new_fc_layer(input = layer_fc1, num_inputs = fc_size, num_outputs = num_classes , use_relu = False)
	y_pred = tf.nn.softmax(layer_fc2)
	y_pred_cls = tf.argmax(y_pred, dimension=1)
	return layer_fc2 , y_pred_cls



	print(accuracy)

def optimizecostfn(x_batch , y_true_batch,ylabels_ce ):
	layer_fc2 , y_pred_cls = compu_graph(x_batch)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer_fc2,labels=ylabels_ce)
	cost = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
	return optimizer , y_pred_cls
	
def next_batch(batch_size):
	global train_images
	global train_labels
	global index_in_epoch
	global epochs_completed
	temp_train = []
	temp_labels = []
	start = index_in_epoch
	index_in_epoch += batch_size
	if(index_in_epoch > num_examples):
		epochs_completed +=1
		perm = np.arange(num_examples)
		np.random.shuffle(perm)
		np.random.shuffle(perm)
		for i in range(len(perm)):
			temp_train.append(np.array(train_images[perm[i]]))
			temp_labels.append(np.array(train_labels[perm[i]]))
		train_images = np.array(temp_train)
		train_labels = np.array(temp_labels)
		start = 0
		index_in_epoch = batch_size
		assert batch_size < num_examples
	end = index_in_epoch
	return train_images[start:end] , train_labels[start:end]
		
	
def run_compu_graph():
	global total_iterations
	create_traindata()
	sess = tf.Session()
	#ximages, ylabels = create_placeholders(batch_size)
	start_time = time.time()
	for i in range(num_iterations):
		ximages, ylabels = next_batch(batch_size)
		x,ximages_pl, y_true_batch,ylabels_ce = create_placeholders(batch_size)
		feed_dict_batch = {x:ximages , ylabels_ce:ylabels}
		optimizer,y_pred_cls = optimizecostfn(ximages_pl,y_true_batch,ylabels_ce)
		sess.run(tf.initialize_all_variables())
		sess.run(optimizer,feed_dict = feed_dict_batch)
		correct_prediction = tf.equal(y_pred_cls, tf.cast(ylabels_ce,tf.int64))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		if(i%20 ==0):
			acc = sess.run(accuracy, feed_dict=feed_dict_batch)
			msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
			print(msg.format(i + 1, acc))
		end_time = time.time()
		time_dif = end_time - start_time
		print("Time usage: " + str(dt.timedelta(seconds=int(round(time_dif)))))
	

run_compu_graph()	
