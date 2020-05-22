import tensorflow as tf

class NN:
    def  makeNN(x,keep_prob):
        # initialization of weight
        def weight_variable(shape):
            # initialization with normal distribution, sigma = 0.1
            inital = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(inital)
        
        # initialization of bias
        def bias_variable(shape):
            # initialization with constant, 0.0
            inital = tf.constant(0.0, shape=shape)
            return tf.Variable(inital)
        
        # conbination layer1
        with tf.name_scope("fc1") as scope:
            W_fc1 = weight_variable([7, 100])   
            b_fc1 = bias_variable([100])
            h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

            # dropout1
            h_fc_1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # conbination layer2
        with tf.name_scope("fc2") as scope:
            W_fc2 = weight_variable([100, 100])   
            b_fc2 = bias_variable([100])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc_1_drop, W_fc2) + b_fc2)

            # dropout2
            h_fc_2_drop = tf.nn.dropout(h_fc2, keep_prob)


        # conbination layer3
        with tf.name_scope("fc3") as scope:
            W_fc3 = weight_variable([100, 100])   
            b_fc3 = bias_variable([100])
            h_fc3 = tf.nn.relu(tf.matmul(h_fc_2_drop, W_fc3) + b_fc3)

            # dropout3
            h_fc_3_drop = tf.nn.dropout(h_fc3, keep_prob)


        # conbination layer4
        with tf.name_scope("fc4") as scope:
            W_fc4 = weight_variable([100, 100])   
            b_fc4 = bias_variable([100])
            h_fc4 = tf.nn.relu(tf.matmul(h_fc_3_drop, W_fc4) + b_fc4)

            # dropout4
            h_fc_4_drop = tf.nn.dropout(h_fc4, keep_prob)

        # conbination layer5
        with tf.name_scope("fc5") as scope:
            W_fc5 = weight_variable([100, 100])   
            b_fc5 = bias_variable([100])
            h_fc5 = tf.nn.relu(tf.matmul(h_fc_4_drop, W_fc5) + b_fc5)

            # dropout5
            h_fc_5_drop = tf.nn.dropout(h_fc5, keep_prob)

        # output layer
        with tf.name_scope("output") as scope:
            W_fc6 = weight_variable([100, 1])
            b_fc6 = bias_variable([1])
            y_ = tf.nn.relu(tf.matmul(h_fc_5_drop, W_fc6) + b_fc6)
   
        return y_