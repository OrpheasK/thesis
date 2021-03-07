#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import csv
import os
import gc
import numpy as np
import tensorflow as tf
from random import randint
from tensorflow import keras
from keras.utils import np_utils, plot_model, to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, concatenate


# In[2]:


# returns train, inference_encoder and inference_decoder models
def define_models(dim_p, n_units):
	# define training encoder
	enc_in_o = Input(shape=(None, 1))
	enc_in_q = Input(shape=(None, 1))
	enc_in_p = Input(shape=(None, dim_p))
	encoder_inputs = concatenate([enc_in_o, enc_in_q, enc_in_p])
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	
	# define training decoder 1
	dec_in_o = Input(shape=(None, 1))
	dec_in_q = Input(shape=(None, 1))
	dec_in_p = Input(shape=(None, dim_p))
	decoder_inputs = concatenate([dec_in_o, dec_in_q, dec_in_p])
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	dec_dense_o = Dense(1, activation='relu', name='tr_out_o')
	dec_dense_q = Dense(1, activation='sigmoid', name='tr_out_q')
	dec_dense_p = Dense(dim_p, activation='softmax', name='tr_out_p')
	#out_o = Dense(1, activation='relu', name='tr_out_o')(decoder_outputs)#act relu
	#out_q = Dense(1, activation='sigmoid', name='tr_out_q')(decoder_outputs)
	#out_p = Dense(dim_p, activation='softmax', name='tr_out_p')(decoder_outputs)
	out_o = dec_dense_o(decoder_outputs)
	out_q = dec_dense_q(decoder_outputs)
	out_p = dec_dense_p(decoder_outputs)

	# define training decoder 2
	dec_in_o_2 = Input(shape=(None, 1))
	dec_in_q_2 = Input(shape=(None, 1))
	dec_in_p_2 = Input(shape=(None, dim_p))
	decoder_inputs_2 = concatenate([dec_in_o_2, dec_in_q_2, dec_in_p_2])
	decoder_lstm_2 = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs_2, _, _ = decoder_lstm_2(decoder_inputs_2, initial_state=encoder_states)
	dec_dense_o_2 = Dense(1, activation='relu', name='tr_out_o_2')
	dec_dense_q_2 = Dense(1, activation='sigmoid', name='tr_out_q_2')
	dec_dense_p_2 = Dense(dim_p, activation='softmax', name='tr_out_p_2')
	#out_o = Dense(1, activation='relu', name='tr_out_o')(decoder_outputs)#act relu
	#out_q = Dense(1, activation='sigmoid', name='tr_out_q')(decoder_outputs)
	#out_p = Dense(dim_p, activation='softmax', name='tr_out_p')(decoder_outputs)
	out_o_2 = dec_dense_o_2(decoder_outputs_2)
	out_q_2 = dec_dense_q_2(decoder_outputs_2)
	out_p_2 = dec_dense_p_2(decoder_outputs_2)
	
	model = Model([enc_in_o, enc_in_q, enc_in_p, dec_in_o, dec_in_q, dec_in_p], [out_o, out_q, out_p])
	model_2 = Model([enc_in_o, enc_in_q, enc_in_p, dec_in_o_2, dec_in_q_2, dec_in_p_2], [out_o_2, out_q_2, out_p_2])
	
	# define inference encoder
	encoder_model = Model([enc_in_o, enc_in_q, enc_in_p], encoder_states)
	
	# define inference decoder 1
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	#out_o = TimeDistributed(Dense(1, activation='relu'))(decoder_outputs)#act relu
	#out_q = TimeDistributed(Dense(1, activation='sigmoid'))(decoder_outputs)
	#out_p = TimeDistributed(Dense(dim_p, activation='softmax'))(decoder_outputs)
	out_o = dec_dense_o(decoder_outputs)
	out_q = dec_dense_q(decoder_outputs)
	out_p = dec_dense_p(decoder_outputs)

	# define inference decoder 2
	decoder_state_input_h_2 = Input(shape=(n_units,))
	decoder_state_input_c_2 = Input(shape=(n_units,))
	decoder_states_inputs_2 = [decoder_state_input_h, decoder_state_input_c_2]
	decoder_outputs_2, state_h_2, state_c_2 = decoder_lstm_2(decoder_inputs_2, initial_state=decoder_states_inputs_2)
	decoder_states_2 = [state_h_2, state_c_2]
	#out_o = TimeDistributed(Dense(1, activation='relu'))(decoder_outputs)#act relu
	#out_q = TimeDistributed(Dense(1, activation='sigmoid'))(decoder_outputs)
	#out_p = TimeDistributed(Dense(dim_p, activation='softmax'))(decoder_outputs)
	out_o_2 = dec_dense_o_2(decoder_outputs_2)
	out_q_2 = dec_dense_q_2(decoder_outputs_2)
	out_p_2 = dec_dense_p_2(decoder_outputs_2)

	decoder_model = Model([dec_in_o, dec_in_q, dec_in_p] + decoder_states_inputs, [out_o, out_q, out_p] + decoder_states)
	decoder_model_2 = Model([dec_in_o_2, dec_in_q_2, dec_in_p_2] + decoder_states_inputs_2, [out_o_2, out_q_2, out_p_2] + decoder_states_2)

	# return all models
	return model, model_2, encoder_model, decoder_model, decoder_model_2


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

#generate train inputs and outputs while one hot encoding pitch and padding for seq2seq
def generatorex(features1, features2, features3, seq_length, batch_size):
    # Create empty arrays to contain batch of features and labels# 
    batch_features1 = np.zeros((batch_size, seq_length, 1))
    batch_features2 = np.zeros((batch_size, seq_length, 1))
    batch_features3 = np.zeros((batch_size, seq_length, 128))
    batch_feat_pad1 = np.zeros((batch_size, seq_length, 1))
    batch_feat_pad2 = np.zeros((batch_size, seq_length, 1))
    batch_feat_pad3 = np.zeros((batch_size, seq_length, 128))
    i = 0
    while True:
        for b in range(batch_size):
            batch_features1[b] = features1[i]
            batch_features2[b] = features2[i]
            batch_features3[b] = to_categorical(features3[i], num_classes=128)
            batch_feat_pad1[b] = np.append([-1], features1[i][:-1]).reshape(seq_length, 1)
            batch_feat_pad2[b] = np.append([-1], features2[i][:-1]).reshape(seq_length, 1)
            batch_feat_pad3[b] = to_categorical(np.append([0], features3[i][:-1]).reshape(seq_length, 1), num_classes=128)
            i += 1
            if (i == len(features1)):
                i=0
        #print(batch_features, batch_labels)
        yield [batch_features1, batch_features2, batch_features3, batch_feat_pad1, batch_feat_pad2, batch_feat_pad3], [batch_features1, batch_features2, batch_features3]


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


# In[10]:


# configure problem
n_features = 127+1
n_steps_out = seq_length
# define models
train, train_2, infenc, infdec, infdec_2 = define_models(n_features, 256)
train.compile(optimizer='adam', loss={'tr_out_o': 'mse', 'tr_out_q': 'mse', 'tr_out_p': 'categorical_crossentropy'},
 metrics={'tr_out_o': 'mean_squared_error', 'tr_out_q': 'mean_squared_error', 'tr_out_p': 'accuracy'})
train_2.compile(optimizer='adam', loss={'tr_out_o_2': 'mse', 'tr_out_q_2': 'mse', 'tr_out_p_2': 'categorical_crossentropy'},
 metrics={'tr_out_o_2': 'mean_squared_error', 'tr_out_q_2': 'mean_squared_error', 'tr_out_p_2': 'accuracy'})


# In[ ]:


# train the two models with alternating epochs
epochs_c = 80
for i in range(epochs_c):
    print ("Epoch", str(i+1)+"_a")
    train.fit(generatorex(dataX1_o, dataX1_q, dataX1_p, seq_length, batch_size=900), steps_per_epoch= (dtlngth[0]-split_i[0]) // 900, verbose=2)
    print ("Epoch", str(i+1)+"_b")
    train_2.fit(generatorex(dataX1_o_2, dataX1_q_2, dataX1_p_2, seq_length, batch_size=900), steps_per_epoch= (dtlngth[1]-split_i[1]) // 900, verbose=2)


# In[11]:


train.save("/data/data1/users/el13102/train.h5")
train_2.save("/data/data1/users/el13102/train_2.h5")
infenc.save("/data/data1/users/el13102/infenc.h5")
infdec.save("/data/data1/users/el13102/infdec.h5")
infdec_2.save("/data/data1/users/el13102/infdec_2.h5")


# In[ ]:




