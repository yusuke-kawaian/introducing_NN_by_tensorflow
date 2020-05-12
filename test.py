import tensorflow as tf
import pandas as pd
import numpy as np

print('Input data....')
INPUT = "normed_dataset.txt"
raw_dataset = pd.read_csv(INPUT, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
raw_dataset = raw_dataset.drop("ion", axis=1)
raw_dataset = raw_dataset.drop("Z", axis=1)
print('complete')

print('Arrange datas....')
dataset = raw_dataset.copy()
dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

drop_col_labels = ['Ptotal']
train_stats = train_dataset.describe()
train_stats = train_stats.drop(drop_col_labels, axis=1)
train_stats = train_stats.transpose()

drop_col_data = ['mass', 'valent', 'pore_d', 'vol', 'r1', 'r2', 'gr_max']
train_labels = train_dataset.drop(drop_col_data, axis=1)
test_labels = test_dataset.drop(drop_col_data, axis=1)

normed_train_dataset = train_dataset.drop(drop_col_labels, axis=1)
normed_test_dataset = test_dataset.drop(drop_col_labels, axis=1)
print('complete')

print('define model')
sess = tf.InteractiveSession()

x = tf.placeholder("float", [None, 7])
y_ = tf.placeholder("float", [None, 1])

W_fc1 = tf.Variable(tf.truncated_normal([7,100], stddev=0.1), name="w1")
b_fc1 = tf.Variable(tf.constant(1.0, shape=[100]), name="b1")
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

W_fc2 = tf.Variable(tf.truncated_normal([100,100], stddev=0.1), name="w2")
b_fc2 = tf.Variable(tf.constant(1.0, shape=[100]), name="b2")
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = tf.Variable(tf.truncated_normal([100,1], stddev=0.1), name="w3")
b_fc3 = tf.Variable(tf.constant(1.0, shape=[1]), name="b3")
y = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_, logits=y))
loss = tf.reduce_sum(tf.square(y - y_))
train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
print('complete')

print('train model')
sess .run(tf. initialize_all_variables ())

for i in range (100000): 
    if i % 1000 == 0:
        feed_dict = {x:normed_train_dataset, y_:train_labels}
        train_loss = sess.run(loss, feed_dict=feed_dict)
        train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
        print("step %d, training accuracy %g, loss %g" % (i, train_accuracy, train_loss))
    train_step.run(feed_dict=feed_dict)

feed_dict = {x:normed_test_dataset, y_:test_labels}
test_loss = sess.run(loss, feed_dict=feed_dict)
test_accuracy = sess.run(accuracy, feed_dict=feed_dict)
print("test accuracy %g" % (test_accuracy))

saver = tf.train.Saver()
saver.save(sess,'./nn_7.100.100_nodropout')


sess = tf.Session()
saver = tf.train.import_meta_graph('nn_7.100.100_nodropout.meta')

saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
W_fc1 = graph.get_tensor_by_name("w1:0")
b_fc1 = graph.get_tensor_by_name("b1:0")
W_fc2 = graph.get_tensor_by_name("w2:0")
b_fc2 = graph.get_tensor_by_name("b2:0")
W_fc3 = graph.get_tensor_by_name("w3:0")
b_fc3 = graph.get_tensor_by_name("b3:0")

x = normed_test_data 
x =tf.cast(x, tf.float32) 

h_fc1 = sess.run(tf.nn.relu(tf.matmul(x, sess.run(W_fc1)) + sess.run(b_fc1)))
h_fc2 = sess.run(tf.nn.relu(tf.matmul(h_fc1, sess.run(W_fc2)) + sess.run(b_fc2)))
y = sess.run(tf.nn.relu(tf.matmul(h_fc2, sess.run(W_fc3)) + sess.run(b_fc3)))

print(y)
print(test_labels)
