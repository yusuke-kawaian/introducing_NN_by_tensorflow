print('NN_predict.py START')

import tensorflow as tf
import datetime
import pandas as pd
import numpy as np
import NN_model as NN

if __name__ == "__main__":
    # https://www.tensorflow.org/api_docs/python/tf/reset_default_graph
    tf.reset_default_graph()

    print('START reproduct model')
    x = tf.placeholder("float", [None, 7])
    y_ = tf.placeholder("float", [None, 1])
    keep_prob = tf.placeholder("float") 

    # definition NN model
    model = NN.NN.makeNN(x, keep_prob) 

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()

    # initialization of variables
    sess.run(tf.global_variables_initializer())
    print('END reproduct model')

    print('Restore Param Start')
    ckpt = tf.train.get_checkpoint_state('./')
    if ckpt: # checkpointがある場吁E
        last_model = ckpt.model_checkpoint_path # pass to model
        print ("Restore load:" + last_model)
        saver.restore(sess, last_model) # input variable datas
    else:
        print('Restore Failed')
    print('Restore Param End')

    # input new dataset
    print('input data....')
    INPUT = "dataset.txt"
    raw_dataset = pd.read_csv(INPUT, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
    dataset = raw_dataset.drop(['ion', 'Z', 'Ptotal'], axis=1)

    print('arrange datas....')
    dataset = dataset.copy()
    dataset = dataset.dropna()

    norm_value = pd.read_csv("normalization.txt", na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
    norm_value = norm_value.drop(['ion', 'Z'], axis=1)
    norm_min = norm_value[0:1]
    norm_max = norm_value[1:2]

    dataset = dataset.astype('float')
    normed_data = (dataset - np.array(norm_min))/np.array(norm_max - np.array(norm_min))

    print('complete')

    # prediction
    pred = pd.DataFrame(model.eval(feed_dict={x:normed_data, keep_prob: 1.0}))
    part_data = raw_dataset.drop(['Z', 'valent', 'r1', 'r2', 'gr_max', 'Ptotal'], axis=1)
    results = pd.concat([part_data, pred], axis=1)
    print('The results are below.')
    print(results)
    results.to_csv("test.csv")

    sess.close()
    print('NN_predict.py END')