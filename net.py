import tensorflow as tf
import utils

tf.set_random_seed(utils.config['model']['image_width'])

X = tf.placeholder(tf.float32, [None, utils.config['model']['image_width'], utils.config['model']['image_height'], 1])
Y_ = tf.placeholder(tf.float32, [None, len(utils.labels)])
pkeep = tf.placeholder(tf.float32)

#Layers
C1 = 4  # first convolutional layer output depth
C2 = 8  # second convolutional layer output depth
C3 = 16 # third convolutional layer output depth
FC4 = 256  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, C1], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([C1], stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([3, 3, C1, C2], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([C2], stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([3, 3, C2, C3], stddev=0.1))
b3 = tf.Variable(tf.truncated_normal([C3], stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([utils.config['model']['image_width']*utils.config['model']['image_height'], FC4], stddev=0.1))
b4 = tf.Variable(tf.truncated_normal([FC4], stddev=0.1))

W5 = tf.Variable(tf.truncated_normal([FC4, len(utils.labels)], stddev=0.1))
b5 = tf.Variable(tf.truncated_normal([len(utils.labels)], stddev=0.1))

stride = 1
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + b1)

k = 2
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + b2)
Y2 = tf.nn.max_pool(Y2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + b3)
Y3 = tf.nn.max_pool(Y3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

#Reshape convolution layers result to fully connected
YY = tf.reshape(Y3, shape=[-1, utils.config['model']['image_width']*utils.config['model']['image_height']])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + b4)
Ylogits = tf.matmul(Y4, W5) + b5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100


#Accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(utils.config['model']['learning_rate']).minimize(cross_entropy)
