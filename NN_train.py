print('NN_train.py START')

import tensorflow as tf
import datetime
import pandas as pd
import numpy as np
import NN_model as NN

#from sklearn.metrics import r2_score

print('Input data....')
INPUT = "dataset.txt"
#column_names = ['ion', 'Z', 'mass', 'valent', 'pore_d', 'vol', 'r1', 'r2', 'gr_max', 'P_total', ''] 
raw_dataset = pd.read_csv(INPUT, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
raw_dataset = raw_dataset.drop(['ion', 'Z'], axis=1)
print('complete')

print('Arrange datas....')
dataset = raw_dataset.copy()
dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.75,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

drop_col_data = ['mass', 'valent', 'pore_d', 'vol', 'r1', 'r2', 'gr_max']
train_labels = train_dataset.drop(drop_col_data, axis=1)*100
test_labels = test_dataset.drop(drop_col_data, axis=1)*100

norm_value = pd.read_csv("normalization.txt", na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
norm_value = norm_value.drop(['ion', 'Z'], axis=1)
norm_min = norm_value[0:1]
norm_max = norm_value[1:2]

drop_col_labels = ['Ptotal']
train_dataset = train_dataset.drop(drop_col_labels, axis=1)
train_dataset = train_dataset.astype('float')
normed_train_data = (train_dataset - np.array(norm_min))/np.array(norm_max - np.array(norm_min))

test_dataset = test_dataset.drop(drop_col_labels, axis=1)
test_dataset = test_dataset.astype('float')
normed_test_data = (test_dataset - np.array(norm_min))/np.array(norm_max - np.array(norm_min))
print('complete')



def loss1(y, y_):
    with tf.name_scope("calculate_RMSE") as scope:
       #loss = -tf.reduce_sum(labels*tf.log(logits)) #closs entropy
       #loss = tf.reduce_mean(tf.square(y - y_))
       rmse = tf.sqrt(tf.losses.mean_squared_error(labels = y_, predictions = y))
       return rmse

def loss2(y, y_):
    with tf.name_scope("calculate_MAE") as scope:
       mae = tf.reduce_mean(tf.abs(y_ - y))
       return mae

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step


if __name__=="__main__":
    with tf.Graph().as_default():
        print('START constructing model') 

        with tf.name_scope("input") as scope:
           x = tf.placeholder("float", [None, 7])

        with tf.name_scope("pred_P") as scope:
           y_ = tf.placeholder("float", [None, 1])

        with tf.name_scope("dropout") as scope:
           keep_prob = tf.placeholder("float")    #dropout

        # definition NN model
        y = NN.NN.makeNN(x, keep_prob) 

        # definition op
        rmse = loss1(y, y_) 
        mae = loss2(y, y_)
        train_op = training(rmse,1e-5) 
        #accur = accuracy(y, y_) 

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # definition variables for TensorBoard
        rmse_op_train = tf.summary.scalar("RMSE on Train", rmse)
        rmse_op_test = tf.summary.scalar("RMSE on Test", rmse)
        mae_op_train = tf.summary.scalar("MAE on Train", mae)
        mae_op_test = tf.summary.scalar("MAE on Test", mae)
        summary_rmse_op_train = tf.summary.merge([rmse_op_train])
        summary_rmse_op_test = tf.summary.merge([rmse_op_test])
        summary_mae_op_train = tf.summary.merge([mae_op_train])
        summary_mae_op_test = tf.summary.merge([mae_op_test])
        summary_writer = tf.summary.FileWriter("./TensorBoard", graph=sess.graph)

        # saving model
        saver = tf.train.Saver()
        print('completing defining model')

        # start learning
        print('learning START : ' + str(datetime.datetime.now()))
        for i in range (1000000): 
            if i % 1000 == 0:
                train_rmse = sess.run(rmse, feed_dict={x:normed_train_data,y_:train_labels, keep_prob: 1.0})
                test_rmse = sess.run(rmse, feed_dict={x:normed_test_data,y_:test_labels, keep_prob: 1.0})
                train_mae = sess.run(mae, feed_dict={x:normed_train_data,y_:train_labels, keep_prob: 1.0})
                test_mae = sess.run(mae, feed_dict={x:normed_test_data,y_:test_labels, keep_prob: 1.0})
                print("step: %d, train_rmse: %g  train_mae: %g test_rmse: %g test_mae: %g"%(i, train_rmse, train_mae, test_rmse, test_mae))
                #print("step: %d, train_rmse: %g train_mae: %g"%(i, train_rmse, train_mae))

                summary_str_rmse_train = sess.run(summary_rmse_op_train, feed_dict={x:normed_train_data,y_:train_labels, keep_prob: 1.0})
                summary_writer.add_summary(summary_str_rmse_train, i)
                summary_str_rmse_test = sess.run(summary_rmse_op_test, feed_dict={x:normed_test_data,y_:test_labels, keep_prob: 1.0})
                summary_writer.add_summary(summary_str_rmse_test, i)
                summary_str_mae_train = sess.run(summary_mae_op_train, feed_dict={x:normed_train_data,y_:train_labels, keep_prob: 1.0})
                summary_writer.add_summary(summary_str_mae_train, i)
                summary_str_mae_test = sess.run(summary_mae_op_test, feed_dict={x:normed_test_data,y_:test_labels, keep_prob: 1.0})
                summary_writer.add_summary(summary_str_mae_test, i)
                summary_writer.flush()
            
            sess.run(train_op, feed_dict={x:normed_train_data,y_:train_labels, keep_prob:0.5})

        print('learning END : ' + str(datetime.datetime.now()))
        #print("train_RMSE: %g  train_MAE: %g test_RMSE: %g test_MAE: %g"%(i, train_rmse, train_mae, test_rmse, test_mae))

        save_path = saver.save(sess,'./nn_model')  
        print('Save END : ' + save_path )

        summary_writer.close()
        sess.close()

        print('NN_train.py END')