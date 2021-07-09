#!/usr/bin/env python
# coding: utf-8

# In[23]:


import csv
import os
import numpy as np
from random import randint
from math import trunc
import pickle
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, concatenate

# In[2]:


# generate target given source sequence
def predict_sequence(infenc, infdec, src_o, src_q, src_p, n_steps, cardinality_o, cardinality_q, cardinality_p):
	# encode
	state = infenc.predict([src_o, src_q, src_p])
	# start of sequence input
	target_o = np.array([0.0 for _ in range(cardinality_o)]).reshape(1, 1, cardinality_o)
	target_q = np.array([0.0 for _ in range(cardinality_q)]).reshape(1, 1, cardinality_q)
	#target_p = 0
	target_p = np.array([0.0 for _ in range(cardinality_p)]).reshape(1, 1, cardinality_p)
	target_o[0][0] = to_categorical([cardinality_o-1], num_classes=cardinality_o)
	target_q[0][0] = to_categorical([cardinality_q-1], num_classes=cardinality_q)
	target_p[0][0] = to_categorical([128], num_classes=cardinality_p)

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
	return output

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [np.argmax(vector) for vector in encoded_seq]

#create list with window length sequences of list a data
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

#create directory for offset, quarterlength one hot encoding
def qldir(m_o, qi):
    fullql = []
    for i in range(0, int(m_o), 1):
        #j = str(i)   
        fullql.append(i+.0) 
        fullql.append(i+.25)
        fullql.append(i+.33)
        fullql.append(i+.5)
        fullql.append(i+.66)
        fullql.append(i+.75)
    fullql.append(m_o)
    fullql.append(m_o+.25)
    fullql[10] = 1.66
    
    if (qi == True):
        r_dict = dict((c, i) for i, c in enumerate(fullql))
    else:
        r_dict = dict((i, c) for i, c in enumerate(fullql))
    
    return r_dict

#generate train inputs and outputs while one hot encoding pitch and padding for seq2seq
def generatorex(features1, features2, features3, seq_length, ft_o, ft_q, ft_p, m_o, m_q, batch_size):
    # Create empty arrays to contain batch of features and labels# 
    q_to_i = qldir(m_o, True)
    batch_features1 = np.zeros((batch_size, seq_length, ft_o))
    batch_features2 = np.zeros((batch_size, seq_length, ft_q))
    batch_features3 = np.zeros((batch_size, seq_length, ft_p))
    batch_feat_pad1 = np.zeros((batch_size, seq_length, ft_o))
    batch_feat_pad2 = np.zeros((batch_size, seq_length, ft_q))
    batch_feat_pad3 = np.zeros((batch_size, seq_length, ft_p))
    i = 0
    while True:
        for b in range(batch_size):
            batch_features1[b] = to_categorical([q_to_i[ql[0]] for ql in features1[i]], num_classes=ft_o)
            batch_features2[b] = to_categorical([q_to_i[ql[0]] for ql in features2[i]], num_classes=ft_q)
            batch_features3[b] = to_categorical(features3[i], num_classes=ft_p)
            batch_feat_pad1[b] = to_categorical(np.append([ft_o-1], [q_to_i[ql[0]] for ql in features1[i][:-1]]).reshape(seq_length, 1), num_classes=ft_o)
            batch_feat_pad2[b] = to_categorical(np.append([ft_q-1], [q_to_i[ql[0]] for ql in features2[i][:-1]]).reshape(seq_length, 1), num_classes=ft_q)
            batch_feat_pad3[b] = to_categorical(np.append([128], features3[i][:-1]).reshape(seq_length, 1), num_classes=ft_p)
            i += 1
            if (i == len(features1)):
                i=0
        #print(batch_features, batch_labels)
        yield [batch_features1, batch_features2, batch_features3, batch_feat_pad1, batch_feat_pad2, batch_feat_pad3], [batch_features1, batch_features2, batch_features3]


# In[4]:

def midiprod(mftxt, fname, infenc, infdec, infdec_2):
	#load data
	stream_list = []

	with open(mftxt, 'r') as f: 
		reader = csv.reader(f)
		sub_list = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]
		stream_list = stream_list + sub_list



	# In[5]:


	#create seperate data structures for each variable (offset, quarterlength, pitch)
	#normalise offset and quarterlength
	offs = []
	qlngth = []
	ptch = []


	max_o = 600.0
	max_q = 50.0

	offsb = max(element[0] for element in stream_list if element[0]<=max_o)
	qlngthb = max(element[1] for element in stream_list if element[1]<=max_q)
	#ptchb = 127.0

	for row in stream_list:
		if (row[0] <= max_o and row[1] <= max_q):
			offs.append(trunc(row[0]*100)/100)
			qlngth.append(trunc(row[1]*100)/100)
			ptch.append(row[2])


	# In[10]:


	#divide the sets in sequences of specific length 
	dtlngth=len(offs)
	n_features_o = int(max_o)*6+2
	n_features_q = int(max_q)*6+2
	n_features_p = 128+1
	seq_length = 30#100 groups of 3

	dataX1_o = rolling_window(np.asarray(offs), seq_length)
	dataX1_q = rolling_window(np.asarray(qlngth), seq_length)
	dataX1_p = rolling_window(np.asarray(ptch), seq_length)

	n_patterns = len(dataX1_p)
	print ("Total Patterns: ", n_patterns)


	# In[11]:


	#reshape inputs to be [samples, time steps, features]
	dataX1_o = np.reshape(dataX1_o, (dtlngth - seq_length + 1, seq_length, 1))
	dataX1_q = np.reshape(dataX1_q, (dtlngth - seq_length + 1, seq_length, 1))
	dataX1_p = np.reshape(dataX1_p, (dtlngth - seq_length + 1, seq_length, 1))




	# In[13]:


	# configure problem
	n_steps_out = seq_length

	# spot check some examples
	re_stream_list = []
	ql_to_int = qldir(max_o, True)
	int_to_ql = qldir(max_o, False)
	#for i in range(n_patterns):
	#    X1_o = np.reshape(to_categorical([ql_to_int[ql[0]] for ql in dataX1_o[i]], num_classes=n_features_o), (1, seq_length, n_features_o))
	#    X1_q = np.reshape(to_categorical([ql_to_int[ql[0]] for ql in dataX1_q[i]], num_classes=n_features_q), (1, seq_length, n_features_q))
	#    X1_p = np.reshape(to_categorical(dataX1_p[i], num_classes=n_features_p), (1, seq_length, n_features_p))
	#    target = predict_sequence(infenc, infdec, X1_o, X1_q, X1_p, n_steps_out, n_features_o, n_features_q, n_features_p)
	#    if (i==0):
	#        for j in range(seq_length):
	#            re_sub_list = [int_to_ql[one_hot_decode([target[3*j]])[0]], int_to_ql[one_hot_decode([target[3*j+1]])[0]], one_hot_decode([target[3*j+2]])]
	#            re_stream_list = re_stream_list + [re_sub_list]
	#    else:
	#        re_sub_list = [int_to_ql[one_hot_decode([target[3*(seq_length-1)]])[0]], int_to_ql[one_hot_decode([target[3*(seq_length-1)+1]])[0]], one_hot_decode([target[3*(seq_length-1)+2]])]
	#        re_stream_list = re_stream_list + [re_sub_list]
	for i in range(0, n_patterns, seq_length):
		X1_o = np.reshape(to_categorical([ql_to_int[ql[0]] for ql in dataX1_o[i]], num_classes=n_features_o), (1, seq_length, n_features_o))
		X1_q = np.reshape(to_categorical([ql_to_int[ql[0]] for ql in dataX1_q[i]], num_classes=n_features_q), (1, seq_length, n_features_q))
		X1_p = np.reshape(to_categorical(dataX1_p[i], num_classes=n_features_p), (1, seq_length, n_features_p))
		target = predict_sequence(infenc, infdec, X1_o, X1_q, X1_p, n_steps_out, n_features_o, n_features_q, n_features_p)
		for j in range(seq_length):
			re_sub_list = [int_to_ql[one_hot_decode([target[3*j]])[0]], int_to_ql[one_hot_decode([target[3*j+1]])[0]], one_hot_decode([target[3*j+2]])]
			re_stream_list = re_stream_list + [re_sub_list]

	# with open("/data/data1/users/el13102/prod.txt","w") as f:
		# wr = csv.writer(f)
		# wr.writerows(map(lambda x: [x], re_stream_list))


	with open("/data/data1/users/el13102/" + fname[:-4] + "_rtj.txt", 'w') as outcsv:   
	   #configure writer to write standard csv file
		writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
	#    writer.writerow(['number', 'number', 'number'])
		for item in re_stream_list:
			#Write item to outcsv
			writer.writerow([item[0], item[1], item[2][0]])
	#with open("/data/data1/users/el13102/prodnew.txt", "wb") as fp:
	#    pickle.dump(re_stream_list, fp)

n_units = 128
# In[17]:


train = load_model("/data/data1/users/el13102/weight/Rock-Jazz/ohe 100x2-30-540/train.h5")
train_2 = load_model("/data/data1/users/el13102/weight/Rock-Jazz/ohe 100x2-30-540/train_2.h5")

encoder_inputs_o = train.input[0]   # input_1 concat
encoder_inputs_q = train.input[1]
encoder_inputs_p = train.input[2]
encoder_outputs, state_h_enc, state_c_enc = train.layers[8].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]

infenc = Model([encoder_inputs_o, encoder_inputs_q, encoder_inputs_p], encoder_states)

decoder_inputs_o = train.input[3]   # input_2 concat
decoder_inputs_q = train.input[4]
decoder_inputs_p = train.input[5]
decoder_inputs = train.layers[7].output
decoder_state_input_h = Input(shape=(n_units,), name='input_20')
decoder_state_input_c = Input(shape=(n_units,), name='input_21')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = train.layers[9]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense_o = train.layers[-3]
decoder_dense_q = train.layers[-2]
decoder_dense_p = train.layers[-1]
decoder_outputs_o = decoder_dense_o(decoder_outputs)
decoder_outputs_q = decoder_dense_q(decoder_outputs)
decoder_outputs_p = decoder_dense_p(decoder_outputs)

infdec = Model([decoder_inputs_o, decoder_inputs_q, decoder_inputs_p] + decoder_states_inputs, [decoder_outputs_o, decoder_outputs_q, decoder_outputs_p] + decoder_states)

decoder_inputs_o_2 = train_2.input[3]   # input_2 concat
decoder_inputs_q_2 = train_2.input[4] 
decoder_inputs_p_2 = train_2.input[5] 
decoder_inputs_2 = train_2.layers[7].output
decoder_state_input_h_2 = Input(shape=(n_units,), name='input_22')
decoder_state_input_c_2 = Input(shape=(n_units,), name='input_23')
decoder_states_inputs_2 = [decoder_state_input_h_2, decoder_state_input_c_2]
decoder_lstm_2 = train_2.layers[9]
decoder_outputs_2, state_h_dec_2, state_c_dec_2 = decoder_lstm_2(decoder_inputs_2, initial_state=decoder_states_inputs_2)
decoder_states_2 = [state_h_dec_2, state_c_dec_2]
decoder_dense_o_2 = train_2.layers[-3]
decoder_dense_q_2 = train_2.layers[-2]
decoder_dense_p_2 = train_2.layers[-1]
decoder_outputs_o_2 = decoder_dense_o_2(decoder_outputs_2)
decoder_outputs_q_2 = decoder_dense_q_2(decoder_outputs_2)
decoder_outputs_p_2 = decoder_dense_p_2(decoder_outputs_2)

infdec_2 = Model([decoder_inputs_o_2, decoder_inputs_q_2, decoder_inputs_p_2] + decoder_states_inputs_2, [decoder_outputs_o_2, decoder_outputs_q_2, decoder_outputs_p_2] + decoder_states_2)



for directory, subdirectories, files in os.walk('C:/Users/Papias/Desktop/thesis/copy/midi/Rock_Cleansed/3'):
	for file in files:
		midiprod(os.path.abspath(os.path.join(directory, file)), file, infenc, infdec, infdec_2)
		