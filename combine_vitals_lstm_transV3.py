

import os
import numpy as np
import random
import tensorflow as tf
import csv
#from modeling import attention_layer
#from modeling import transformer_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


#from lstm_train import train_model
#from lstm_train_v3 import train_model
from lstm_transV3_train import train_model


'''This is the code for combining vitals into a LSTM-transformer score'''



def read_head(data):
	head=data[0]
	head_num_dic={}
	for i in range(len(head)):
		head_num_dic[head[i]]=i
	return head_num_dic


def to_one_hot(k,n=2):
	ls=[0 for i in range(n)]
	ls[k]=1
	return ls


def normalize(data):
	#data=np.array(data)
	scaler = preprocessing.StandardScaler()
	data = scaler.fit_transform(data)
	#data=data.tolist()
	return data


def df_drop(df_new, drop_cols):
    return df_new.drop(df_new.columns[df_new.columns.isin(drop_cols)], axis=1)

# ------------------ DATA prepare




df = pd.read_csv('data_file.csv',index_col=0)




vital_cols=[]

for v in ['s_bp','d_bp','pulse','resp','spo2','temp']:
	for k in range(1,7):
		vital_cols.append(v+'_'+str(k)+'tau')
		

print(vital_cols)

ALL_DATA=df[vital_cols]
ALL_LABEL=df['flag_ICU']


ALL_DATA=normalize(ALL_DATA) # skip this if already normalized


print(ALL_LABEL.shape)
print(ALL_DATA.shape)



output, auc_ = train_model(X_train_ori=ALL_DATA, xtest=ALL_DATA, y_train_ori=ALL_LABEL, ytest=ALL_LABEL, x_to_be_output=ALL_DATA, seed=1)
output=output.reshape([-1,1])

print(output.shape)
print(output)
df['vitals_lstm']=output


AUC=[]

for s in range(2020,2025):
	X_train_ori, xtest, y_train_ori, ytest = train_test_split(ALL_DATA, ALL_LABEL, stratify=ALL_LABEL, test_size=0.2, random_state=s)
	#print(ytest)
	output,auc_ = train_model(X_train_ori=X_train_ori, xtest=xtest, y_train_ori=y_train_ori, ytest=ytest, x_to_be_output=ALL_DATA, seed=7) 
	output=output.reshape([-1,1])
	df['vitals_lstm_'+str(s)]=output
	AUC.append(auc_)

print(np.mean(AUC))

df=df_drop(df,vital_cols)



df.to_csv('output.csv')




