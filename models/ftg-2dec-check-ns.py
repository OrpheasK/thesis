#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import os
import numpy as np
from random import randint
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Model, load_model


# In[2]:


# generate target given source sequence
def predict_sequence(infenc, infdec, src_o, src_q, src_p, n_steps, cardinality):
	# encode
	state = infenc.predict([src_o, src_q, src_p])
	# start of sequence input
	target_o = np.array([-1]).reshape(1, 1, 1)
	target_q = np.array([-1]).reshape(1, 1, 1)
	#target_p = 0
	target_p = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		#print(target_o.shape)
		#print(target_q.shape)
		#print(target_p.shape)
		#print(state[0].shape)
		o, q, p, h, c = infdec.predict([target_o, target_q, target_p] + state)
		#print(a)
		# store prediction
		output.append(o[0,0,:])
		output.append(q[0,0,:])
		output.append(p[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_o = o
		target_q = q
		target_p = p
	return np.array(output)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [np.argmax(vector) for vector in encoded_seq]

#create list with window length sequences of list a data
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# In[3]:


#load data
stream_list = []
stream_list_2 = []

for path, subdirectories, files in os.walk('/data/data1/users/el13102/midi21txt/Rock_Cleansed/1'):
    for name in files:
        with open(os.path.join(path, name), 'r') as f: 
            reader = csv.reader(f)
            sub_list = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]
            stream_list = stream_list + sub_list
            
for path, subdirectories, files in os.walk('/data/data1/users/el13102/midi21txt/Jazz_Cleansed'):
    for name in files:
        with open(os.path.join(path, name), 'r') as f: 
            reader = csv.reader(f)
            sub_list = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]
            stream_list_2 = stream_list_2 + sub_list


# In[4]:


#create seperate data structures for each variable (offset, quarterlength, pitch)
#normalise offset and quarterlength
offs = []
qlngth = []
ptch = []

offs_2 = []
qlngth_2 = []
ptch_2 = []

offsb = max(element[0] for element in stream_list if element[0]<=600.0)
qlngthb = max(element[1] for element in stream_list if element[1]<=50.0)
#ptchb = 127.0
offsb_2 = max(element[0] for element in stream_list_2 if element[0]<=600.0)
qlngthb_2 = max(element[1] for element in stream_list_2 if element[1]<=50.0)

for row in stream_list:
    if (row[0] <= 600.0 and row[1] <= 50.0):
        offs.append(row[0]/offsb)
        qlngth.append(row[1]/qlngthb)
        ptch.append(row[2])
        
for row in stream_list_2:
    if (row[0] <= 600.0 and row[1] <= 50.0):
        offs_2.append(row[0]/offsb_2)
        qlngth_2.append(row[1]/qlngthb_2)
        ptch_2.append(row[2])


# In[5]:


#divide the sets in sequences of specific length 
dtlngth=[len(offs), len(offs_2)]
seq_length = 20#100 groups of 3

dataX1_o = rolling_window(np.asarray(offs), seq_length)
dataX1_q = rolling_window(np.asarray(qlngth), seq_length)
dataX1_p = rolling_window(np.asarray(ptch), seq_length)

dataX1_o_2 = rolling_window(np.asarray(offs_2), seq_length)
dataX1_q_2 = rolling_window(np.asarray(qlngth_2), seq_length)
dataX1_p_2 = rolling_window(np.asarray(ptch_2), seq_length)

n_patterns = [len(dataX1_p), len(dataX1_p_2)]
print ("Total Patterns: ", n_patterns)


# In[6]:


#reshape inputs to be [samples, time steps, features]
dataX1_o = np.reshape(dataX1_o, (dtlngth[0] - seq_length + 1, seq_length, 1))
dataX1_q = np.reshape(dataX1_q, (dtlngth[0] - seq_length + 1, seq_length, 1))
dataX1_p = np.reshape(dataX1_p, (dtlngth[0] - seq_length + 1, seq_length, 1))

dataX1_o_2 = np.reshape(dataX1_o_2, (dtlngth[1] - seq_length + 1, seq_length, 1))
dataX1_q_2 = np.reshape(dataX1_q_2, (dtlngth[1] - seq_length + 1, seq_length, 1))
dataX1_p_2 = np.reshape(dataX1_p_2, (dtlngth[1] - seq_length + 1, seq_length, 1))


# In[7]:


#divide data in train and validation sets
split_i = [n_patterns[0]*10 // 100, n_patterns[1]*10 // 100]

dataX1_o_v = dataX1_o[-split_i[0]:]
dataX1_o = dataX1_o[:-split_i[0]]

dataX1_q_v = dataX1_q[-split_i[0]:]
dataX1_q = dataX1_q[:-split_i[0]]

dataX1_p_v = dataX1_p[-split_i[0]:]
dataX1_p = dataX1_p[:-split_i[0]]

dataX1_o_v_2 = dataX1_o_2[-split_i[1]:]
dataX1_o_2 = dataX1_o_2[:-split_i[1]]

dataX1_q_v_2 = dataX1_q_2[-split_i[1]:]
dataX1_q_2 = dataX1_q_2[:-split_i[1]]

dataX1_p_v_2 = dataX1_p_2[-split_i[1]:]
dataX1_p_2 = dataX1_p_2[:-split_i[1]]

print ("Validation Patterns: ", split_i)


# In[8]:


# configure problem
n_features = 127+1
n_steps_out = seq_length


# In[9]:


train = load_model("/data/data1/users/el13102/train.h5")
train_2 = load_model("/data/data1/users/el13102/train_2.h5")
infenc = load_model("/data/data1/users/el13102/infenc.h5", compile=False)
infdec = load_model('/data/data1/users/el13102/infdec.h5', compile=False)
infdec_2 = load_model('/data/data1/users/el13102/infdec_2.h5', compile=False)


# In[10]:


# spot check some examples
for _ in range(2):
    i = randint(1, split_i[0])
    X1_o = np.reshape(dataX1_o_v[i], (1, seq_length, 1))
    X1_q = np.reshape(dataX1_q_v[i], (1, seq_length, 1))
    X1_p = np.reshape(to_categorical(dataX1_p_v[i], num_classes=n_features), (1, seq_length, n_features))
    target = predict_sequence(infenc, infdec_2, X1_o, X1_q, X1_p, n_steps_out, n_features)
    for j in range(seq_length):
        print('X_o=%s, y_o=%s, X_q=%s, y_q=%s, X_p=%s, y_p=%s' % (dataX1_o_v[i][j]*offsb, target[3*j]*offsb,
                                                                  dataX1_q_v[i][j]*qlngthb, target[3*j+1]*qlngthb,
                                                                  dataX1_p_v[i][j], one_hot_decode([target[3*j+2]])))
    print()


# In[11]:


# spot check some examples
for _ in range(2):
    i = randint(1, split_i[1])
    X1_o = np.reshape(dataX1_o_v_2[i], (1, seq_length, 1))
    X1_q = np.reshape(dataX1_q_v_2[i], (1, seq_length, 1))
    X1_p = np.reshape(to_categorical(dataX1_p_v_2[i], num_classes=n_features), (1, seq_length, n_features))
    target = predict_sequence(infenc, infdec, X1_o, X1_q, X1_p, n_steps_out, n_features)
    for j in range(seq_length):
        print('X_o=%s, y_o=%s, X_q=%s, y_q=%s, X_p=%s, y_p=%s' % (dataX1_o_v_2[i][j]*offsb, target[3*j]*offsb,
                                                                  dataX1_q_v_2[i][j]*qlngthb, target[3*j+1]*qlngthb,
                                                                  dataX1_p_v_2[i][j], one_hot_decode([target[3*j+2]])))
    print()


# In[ ]:




