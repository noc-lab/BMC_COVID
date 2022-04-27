import os
import numpy as np
import random
import tensorflow as tf
import csv
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

from modeling import attention_layer
from modeling import transformer_model


'''This is the code for combining vitals into a LSTM-transformer score'''


tf.set_random_seed(7)
random.seed(7)

#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='1'





def to_one_hot(k,n=2):
	ls=[0 for i in range(n)]
	ls[k]=1
	return ls


def normalize(data):
	data=np.array(data)
	scaler = preprocessing.StandardScaler()
	data = scaler.fit_transform(data)
	data=data.tolist()
	return data





#--------------------Define model
#tf.reset_default_graph()

x = tf.placeholder("float", shape=[None,36])
y_ = tf.placeholder("float", shape=[None,1])
keep_prob = tf.placeholder(tf.float32)
pos_weight = tf.placeholder(tf.float32)
trans_drop = tf.placeholder("float")

#------------------------------------------------------

x_in=tf.reshape(x, [-1,36])
x_lstm=tf.reshape(x, [-1,6,6])
x_lstm=tf.transpose(x_lstm, perm=[0, 2, 1])


current=x_lstm

#att=tf.reduce_mean(current,1)


#--------------------- hyper parameters

# different models should try different hyperparameters



'''
# mean-impute 0-drop params
dim=32
inter_dim=16
layer=6
heads=4
transformer_dropout=0
lr_=0.0005
use_activation=True
'''


'''
# mean-impute 12-drop params
dim=32
inter_dim=32
layer=5
heads=4
transformer_dropout=0
lr_=0.0002
use_activation=True
'''



# mean-impute 24-drop params
dim=32
inter_dim=64
layer=6
heads=4
transformer_dropout=0
lr_=0.0002
use_activation=True


#---------------------------------------------------------------------------

lstmFwCell2=tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=dim,state_is_tuple=True),output_keep_prob=keep_prob)
lstmBwCell2=tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=dim,state_is_tuple=True),output_keep_prob=keep_prob)

outputs_,output_state=tf.nn.bidirectional_dynamic_rnn(lstmFwCell2,lstmBwCell2,current,dtype=float,scope="bi-lstm2")
fw1=output_state[0][1]
bw1=output_state[1][1]

outputs_=tf.concat(outputs_,2)
#outputs_=tf.split(outputs_,2,-1)
#outputs_=(outputs_[0]+outputs_[1])/2
current=outputs_

if use_activation==True:
	current=tf.nn.relu(current)

current=transformer_model(input_tensor=current,
                      attention_mask=None,
                      hidden_size=dim*2,
                      num_hidden_layers=layer,
                      num_attention_heads=heads,
                      intermediate_size=inter_dim,
                      hidden_dropout_prob=trans_drop,
                      attention_probs_dropout_prob=trans_drop,
                      initializer_range=0.02,
                      do_return_all_layers=False)

print(current.shape)


current=current[:,0,:]
#current=tf.concat([fw1,bw1],1)
output_layer=current

#current=tf.reduce_mean(current,1)

W = tf.Variable(tf.truncated_normal([dim*2, 1] ,stddev=0.1))
B = tf.Variable(tf.zeros([1]))


#y  = tf.matmul(x_in[:,0:42], W)+B
y  = tf.matmul(current, W)+B
#outputs = tf.split(current, 2, -1)


#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y_, logits=y, pos_weight=pos_weight))

#tf.summary.scalar('loss', cross_entropy)
#cross_entropy = tf.reduce_mean(abs(y_-y))
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(lr_).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred=tf.argmax(y, 1)


sess = tf.InteractiveSession()
#merged = tf.summary.merge_all()
tf.global_variables_initializer().run()

saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1000)





def train_model(X_train_ori, xtest, y_train_ori, ytest, x_to_be_output, seed=6):
	
	#y_train_ori=np.array([to_one_hot(i) for i in y_train_ori.tolist()])
	#ytest=np.array([to_one_hot(i) for i in ytest.tolist()])

	y_train_ori=np.array([[i] for i in y_train_ori.tolist()])
	ytest=np.array([[i] for i in ytest.tolist()])
	
	SaveName='tp_save'
	X_train, xval, y_train, yval = train_test_split(X_train_ori, y_train_ori, stratify=y_train_ori, test_size=0.2,  random_state=seed)
	current_best=0
	current_best_epoch=0
	tf.global_variables_initializer().run()

	for epoch in range(500):

		#pos_ratio=np.mean(np.argmax(y_train, 1))
		pos_ratio=np.mean(y_train)
		pos_w=(1-pos_ratio)/pos_ratio
		#pos_w=0.5
		#print(pos_w)
		sess.run(train_step, feed_dict={x:X_train, y_:y_train, keep_prob:0.9, pos_weight:pos_w, trans_drop:transformer_dropout})

		if (epoch+1)%20==0:
		#print(cycle)
			[acc,p,gt]=sess.run([accuracy,tf.math.sigmoid(y),y_], feed_dict ={x:xval, y_:yval, keep_prob:1, trans_drop:0})
			[acc_,p_,gt_]=sess.run([accuracy,tf.math.sigmoid(y),y_], feed_dict ={x:xtest, y_:ytest, keep_prob:1, trans_drop:0})
			prodict_prob_y=p
			prodict_prob_y_=p_
			auc=metrics.roc_auc_score(gt,prodict_prob_y)
			auc_=metrics.roc_auc_score(gt_,prodict_prob_y_)
		
			if auc>current_best:
				current_best=auc
				current_best_epoch=epoch
				save_path = saver.save(sess,SaveName)

			print('Epoch:',epoch, auc, current_best, auc_)
		#print(current_best_epoch)

	saver.restore(sess,SaveName)
	[acc,p,gt]=sess.run([accuracy,tf.math.sigmoid(y),y_], feed_dict ={x:X_train_ori, y_:y_train_ori, keep_prob:1, trans_drop:0})
	[acc_,p_,gt_]=sess.run([accuracy,tf.math.sigmoid(y),y_], feed_dict ={x:xtest, y_:ytest, keep_prob:1, trans_drop:0})
	prodict_prob_y=p
	prodict_prob_y_=p_
	auc=metrics.roc_auc_score(gt,prodict_prob_y)
	auc_=metrics.roc_auc_score(gt_,prodict_prob_y_)
	print('lstm test auc: ', auc_, 'lstm train auc: ', auc, 'lstm validation auc:', current_best)
	
	#p_out=sess.run(tf.nn.softmax(y), feed_dict ={x:x_to_be_output, keep_prob:1})
	p_out=sess.run(y, feed_dict ={x:x_to_be_output, keep_prob:1, trans_drop:0})
	#return prodict_prob_y, prodict_prob_y_
	print(p_out)
	return p_out[:,0], auc_







