# introducing_NN_by_tensorflow
This trial aims introduction of NN to my research by tensorflow of Python. **To keep accuracy of this README, I write on Japanese, README (Japanese).md, too just in case**.    

# Overview
My study investigates the characteristics that metal cations selectivity adsorb to micro porous carbon with applied voltage by MD simulation. In this trial, I construct the ML model that predicts the probability `pred_P` that metal cations adsorb to a pore with 7 parameters, `mass`, `valent`, the first/second hydration radius `r1/ r2`, the maximum value of RDF `gr_max`, voltage `vol` and pore diameter `pore_d`, by R. The number of dataset is **157**.       

# Description  
This trial carried out on anaconda3/5.3.1.  
## Package
* tensorflow 1.12.0 
* pandas 0.25.3  
* numpy 1.14.5   

## constructing NN model
### shaping dataset
The parameters is 7 above, `mass`, `valent`, `r1`, `r2`, `gr_max`, `vol` and `pore_d`.  
The dependent variable is `pred_P` above.  

These parameteres normlized by the below equation in advance.  
```
def norm(x):
    return (x-train_stats['min'])/(train_stats['max']-train_stats['min'])
```

In this trial, I separated 80% data as training dataset by below codes.  
```
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
```

### definition NN model
In this trial, I constructed below NN model by below codes.  

| layer | type | output size | act. func. |    
----|----|----|----  
| input | input | 7 | - |  
| hidden1 | total binding | 100 | ReLU |  
| hidden2 | total binding | 100 | ReLU |  
| output | total binding | 1 | ReLU |  

```
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
```

### loss function
In this trial, I used **MSE** as error function and used **Adam Optimizer**.    
``` 
loss = tf.reduce_sum(tf.square(y - y_))
train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
```

### accurracy
```
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
```

### training model
```
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
```

### saving and reproducting model  
```
saver = tf.train.Saver()
saver.save(sess,'./test')


sess = tf.Session()
saver = tf.train.import_meta_graph('test.meta')

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
```

# Conclusion    
I could construct NN model by tensorflow. However, **the behavior of accuracy shows 1 constantly like below although loss converged**.  
```
step 0, training accuracy 1, loss 201.881
step 1000, training accuracy 1, loss 14.555
step 2000, training accuracy 1, loss 11.1428
step 3000, training accuracy 1, loss 7.86557
step 4000, training accuracy 1, loss 5.43377
step 5000, training accuracy 1, loss 4.0193
step 6000, training accuracy 1, loss 3.29407
step 7000, training accuracy 1, loss 3.09101
step 8000, training accuracy 1, loss 3.04178
step 9000, training accuracy 1, loss 3.00617
step 10000, training accuracy 1, loss 2.92083
```

# My Problems  
* the behavior of accuracy is strange like below although loss converged.  
* which error func. do I use, MSE or closs enthoropy?   
