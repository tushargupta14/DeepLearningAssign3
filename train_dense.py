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

def reshape_test_data(testX,testY) :

    test_vectors = []
    target_vectors = []

    # Change the 2D image into a one 1D vector
    for image in testX:
        image.shape = (784) # 28*28 = 784
        test_vectors.append(image)
    test_vectors = np.array(test_vectors)

    print test_vectors.shape


    for i in range(len(testY)):
        target = np.zeros((10))
        target[testY[i]] = 1
        target_vectors.append(target)
    target_vectors = np.array(target_vectors)

    print target_vectors.shape

    return test_vectors,target_vectors

def train(trainX, trainY):
    

    with tf.Graph().as_default():

        x_train,y_train = reshape_data(trainX,trainY)

        #x_test, y_test = reshape_test_data(testX,testY)

        sess = tf.InteractiveSession()

        x = tf.placeholder(tf.float32,shape = [None,784])
        y = tf.placeholder(tf.float32, shape = [None,10])

        W = tf.Variable(tf.zeros([784,10]),name = "weights")
        b = tf.Variable(tf.zeros([10]), name = "biases")

        sess.run(tf.global_variables_initializer())
        y_predicted = tf.matmul(x,W) + b



        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predicted))

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        

        n_train = x_train.shape[0] # 60000

        batch_size = 200

        counter = 0
        
        saver = tf.train.Saver()
        


        for iterations in range(n_train/batch_size) : 

            x_batch = x_train[counter : counter+batch_size,]
            y_batch = y_train[counter : counter+batch_size, ]
            counter += batch_size

            sess.run(train_step, feed_dict={x: x_batch,y:y_batch})
        save_path = saver.save(sess,"weights/single_mlp_model.ckpt")
        print("Model saved in file: %s" % save_path)



        sess.close()
    """correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: x_test,
                                      y : y_test}))

    prediction=tf.argmax(y_predicted,1)
    print "predictions", prediction.eval(feed_dict={x: x_test}, session=sess)

    """

def test(testX):
    
    x = tf.placeholder(tf.float32,shape = [None,784])
    y = tf.placeholder(tf.float32, shape = [None,10])

    W = tf.Variable(tf.zeros([784,10]),name = "weights")
    b = tf.Variable(tf.zeros([10]), name = "biases")
    y_predicted = tf.matmul(x,W) + b
    
    saver = tf.train.Saver()
    

    test_vectors = []

    # Change the 2D image into a one 1D vector
    for image in testX:
        image.shape = (784) # 28*28 = 784
        test_vectors.append(image)
    test_vectors = np.array(test_vectors)

    print test_vectors.shape
    x_test = test_vectors
    with tf.Session() as sess :

        sess.run(tf.global_variables_initializer())

        saver.restore(sess, "weights/single_mlp_model.ckpt")

       
        prediction=tf.argmax(y_predicted,1)

        #print "predictions", prediction.eval(feed_dict={x: x_test}, session=sess)
        return prediction.eval(feed_dict = { x: x_test} , session = sess)
