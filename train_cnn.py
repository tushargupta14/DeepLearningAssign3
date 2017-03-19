'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Tushar Gupta 
Roll No.: 13CH30023

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf 


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def reshape_data(trainX,trainY) :

    training_vectors = []
    target_vectors = []

    # Change the 2D image into a one 1D vector
    for image in trainX:
        image.shape = (784) # 28*28 = 784
        training_vectors.append(image)
    training_vectors = np.array(training_vectors)

    print training_vectors.shape


    for i in range(len(trainY)):
        target = np.zeros((10))
        target[trainY[i]] = 1
        target_vectors.append(target)
    target_vectors = np.array(target_vectors)
    print target_vectors.shape
    return training_vectors,target_vectors


def train(trainX, trainY):
    
    with tf.Graph().as_default():

        sess = tf.InteractiveSession()
        x_train,y_train = reshape_data(trainX,trainY)

        x = tf.placeholder(tf.float32,shape = [None,784])
        y = tf.placeholder(tf.float32, shape = [None,10])

        W_conv1 = weight_variable([5, 5, 1, 32])    
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1,28,28,1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)


        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

        
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


        sess.run(tf.global_variables_initializer())

        n_train = x_train.shape[0]
        batch_size = 200
        counter = 0

        saver = tf.train.Saver()

        for iterations in range(n_train/batch_size) : 

            x_batch = x_train[counter : counter+batch_size,]
            y_batch = y_train[counter : counter+batch_size, ]
            counter += batch_size

            sess.run(train_step, feed_dict={x: x_batch,y:y_batch, keep_prob : 0.5})


        save_path = saver.save(sess,"weights/cnn_model.ckpt")
        print("Model saved in file: %s" % save_path)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_train, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(sess.run(accuracy, feed_dict={x: x_train,
                                          y : y_train}))

        sess.close()
def test(testX):



    test_vectors = []

    # Change the 2D image into a one 1D vector
    for image in testX:
        image.shape = (784) # 28*28 = 784
        test_vectors.append(image)
    test_vectors = np.array(test_vectors)

    print test_vectors.shape
    x_test = test_vectors

    with tf.Graph().as_default():
        
        x = tf.placeholder(tf.float32,shape = [None,784])
        y = tf.placeholder(tf.float32, shape = [None,10])

        W_conv1 = weight_variable([5, 5, 1, 32])    
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1,28,28,1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)


        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        saver = tf.train.Saver()

        
       

        with tf.Session() as sess :

            sess.run(tf.global_variables_initializer())

            saver.restore(sess, "weights/cnn_model.ckpt")

           
            prediction=tf.argmax(y_conv,1)

            print "predictions", prediction.eval(feed_dict={x: x_test, keep_prob : 0.5}, session=sess)
            return prediction.eval(feed_dict = { x: x_test , keep_prob : 0.5} , session = sess)


