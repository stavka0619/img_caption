import tensorflow as tf
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
from imagenet_classes import class_names
import sys

def weight_variable(shape):
  initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1, name='weights')
  return tf.Variable(initial)

def bias_variable(shape, b):
  initial = tf.constant(b, shape=shape)
  return tf.Variable(initial, trainable=True, name='biases')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# reference:
# https://www.cs.toronto.edu/~frossard/post/vgg16/
class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.parameters = []
        self.conv_layers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3)
        # load weight
        if weights is not None and session is not None:            
            self.load_weight(weights, session)
        
    def conv_layers(self):
        # zero-mean input
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        images =self.imgs-mean
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            W_conv = weight_variable([3,3,3,64])
            biases = bias_variable([64], 0.0)
            conv = conv2d(images, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv1_1 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            W_conv = weight_variable([3,3,64,64])
            biases = bias_variable([64], 0.0)
            conv = conv2d(self.h_conv1_1, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv1_2 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # pool1
        self.h_pool1 = max_pool_2x2(self.h_conv1_2)
        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            W_conv = weight_variable([3,3,64,128])
            biases = bias_variable([128], 0.0)
            conv = conv2d(self.h_pool1, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv2_1 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            W_conv = weight_variable([3,3,128,128])
            biases = bias_variable([128], 0.0)
            conv = conv2d(self.h_conv2_1, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv2_2 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        #pool2
        self.h_pool2 = max_pool_2x2(self.h_conv2_2)
        # conv3_1
        with tf.name_scope('conv3_1') as scope:        
            W_conv = weight_variable([3,3,128,256])
            biases = bias_variable([256], 0.0)
            conv = conv2d(self.h_pool2, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv3_1 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # conv3_2
        with tf.name_scope('conv3_2') as scope:        
            W_conv = weight_variable([3,3,256,256])
            biases = bias_variable([256], 0.0)
            conv = conv2d(self.h_conv3_1, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv3_2 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # conv3_3
        with tf.name_scope('conv3_3') as scope:        
            W_conv = weight_variable([3,3,256,256])
            biases = bias_variable([256], 0.0)
            conv = conv2d(self.h_conv3_2, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv3_3 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # pool3
        self.h_pool3 = max_pool_2x2(self.h_conv3_3)
        # conv4_1
        with tf.name_scope('conv4_1') as scope:        
            W_conv = weight_variable([3,3,256,512])
            biases = bias_variable([512], 0.0)
            conv = conv2d(self.h_pool3, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv4_1 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # conv4_2
        with tf.name_scope('conv4_2') as scope:        
            W_conv = weight_variable([3,3,512,512])
            biases = bias_variable([512], 0.0)
            conv = conv2d(self.h_conv4_1, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv4_2 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # conv4_3
        with tf.name_scope('conv4_3') as scope:        
            W_conv = weight_variable([3,3,512,512])
            biases = bias_variable([512], 0.0)
            conv = conv2d(self.h_conv4_2, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv4_3 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # pool4
        self.h_pool4 = max_pool_2x2(self.h_conv4_3)
        # conv5_1
        with tf.name_scope('conv5_1') as scope:        
            W_conv = weight_variable([3,3,512,512])
            biases = bias_variable([512], 0.0)
            conv = conv2d(self.h_pool4, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv5_1 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # conv5_2
        with tf.name_scope('conv5_2') as scope:        
            W_conv = weight_variable([3,3,512,512])
            biases = bias_variable([512], 0.0)
            conv = conv2d(self.h_conv5_1, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv5_2 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # conv5_3
        with tf.name_scope('conv5_3') as scope:        
            W_conv = weight_variable([3,3,512,512])
            biases = bias_variable([512], 0.0)
            conv = conv2d(self.h_conv5_2, W_conv)
            conv_bias = tf.nn.bias_add(conv, biases)
            self.h_conv5_3 = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [W_conv, biases]
        # pool5
        self.h_pool5 = max_pool_2x2(self.h_conv5_3)
        
    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            n = 7*7*512
            W_fc1 = weight_variable([n,4096])
            biases = bias_variable([4096], 0.0)
            h_pool5_flat = tf.reshape(self.h_pool5, [-1,n])
            h_fc1 = tf.nn.bias_add(tf.matmul(h_pool5_flat, W_fc1), biases)
            self.fc1 = tf.nn.relu(h_fc1)
            self.parameters += [W_fc1, biases]
        # fc2
        with tf.name_scope('fc2') as scope:
            n = 4096
            W_fc2 = weight_variable([n,4096])
            biases = bias_variable([4096], 0.0)
            h_fc2 = tf.nn.bias_add(tf.matmul(self.fc1, W_fc2), biases)
            self.fc2 = tf.nn.relu(h_fc2)
            self.parameters += [W_fc2, biases]
        # fc3
        with tf.name_scope('fc3') as scope:
            n = 4096
            W_fc3 = weight_variable([n,1000])
            biases = bias_variable([1000], 0.0)
            h_fc3 = tf.nn.bias_add(tf.matmul(self.fc2, W_fc3), biases)
            self.fc3 = tf.nn.relu(h_fc3)
            self.parameters += [W_fc3, biases]

    def load_weight(self, weight_file, session):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i,k in enumerate(keys):
            print i, k, np.shape(weights[k])
            session.run(self.parameters[i].assign(weights[k]))
            
if __name__ == '__main__':
    img = imread(sys.argv[1], mode='RGB')
    img = imresize(img, (224,224))
    print 
    session = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, "/home/stavka/Downloads/vgg16_weights.npz", session);

    prob = session.run(vgg.probs, feed_dict={vgg.imgs: [img]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print class_names[p], prob[p]
                     
