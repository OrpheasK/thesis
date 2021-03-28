#!/usr/bin/env python
# coding: utf-8

# In[23]:


import csv
import os
import numpy as np
from random import randint
from math import trunc
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
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
            batch_feat_pad2[b] = to_categorical(np.append([ft_o-1], [q_to_i[ql[0]] for ql in features2[i][:-1]]).reshape(seq_length, 1), num_classes=ft_q)
            batch_feat_pad3[b] = to_categorical(np.append([128], features3[i][:-1]).reshape(seq_length, 1), num_classes=ft_p)
            i += 1
            if (i == len(features1)):
                i=0
        #print(batch_features, batch_labels)
        yield [batch_features1, batch_features2, batch_features3]


# In[4]:


#load data
stream_list = []
stream_list_2 = []

for path, subdirectories, files in os.walk('/data/data1/users/el13102/midi21txt/Rock_Cleansed/678'):
    for name in files:
        with open(os.path.join(path, name), 'r') as f: 
            reader = csv.reader(f)
            sub_list = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]
            stream_list = stream_list + sub_list
            
for path, subdirectories, files in os.walk('/data/data1/users/el13102/midi21txt/lastfm/jazz_cleansed'):
    for name in files:
        with open(os.path.join(path, name), 'r') as f: 
            reader = csv.reader(f)
            sub_list = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]
            stream_list_2 = stream_list_2 + sub_list


# In[5]:


#create seperate data structures for each variable (offset, quarterlength, pitch)
#normalise offset and quarterlength
offs = []
qlngth = []
ptch = []

offs_2 = []
qlngth_2 = []
ptch_2 = []

max_o = 600.0
max_q = 50.0

offsb = max(element[0] for element in stream_list if element[0]<=max_o)
qlngthb = max(element[1] for element in stream_list if element[1]<=max_q)
#ptchb = 127.0
offsb_2 = max(element[0] for element in stream_list_2 if element[0]<=max_o)
qlngthb_2 = max(element[1] for element in stream_list_2 if element[1]<=max_q)

for row in stream_list:
    if (row[0] <= max_o and row[1] <= max_q):
        offs.append(trunc(row[0]*100)/100)
        qlngth.append(trunc(row[1]*100)/100)
        ptch.append(row[2])
        
for row in stream_list_2:
    if (row[0] <= max_o and row[1] <= max_q):
        offs_2.append(trunc(row[0]*100)/100)
        qlngth_2.append(trunc(row[1]*100)/100)
        ptch_2.append(row[2])


# In[10]:


#divide the sets in sequences of specific length 
dtlngth=[len(offs), len(offs_2)]
n_features_o = int(max_o)*6+2
n_features_q = int(max_q)*6+2
n_features_p = 128+1
seq_length = 30#100 groups of 3
dataX1_o = rolling_window(np.asarray(offs), seq_length)
dataX1_q = rolling_window(np.asarray(qlngth), seq_length)
dataX1_p = rolling_window(np.asarray(ptch), seq_length)

dataX1_o_2 = rolling_window(np.asarray(offs_2), seq_length)
dataX1_q_2 = rolling_window(np.asarray(qlngth_2), seq_length)
dataX1_p_2 = rolling_window(np.asarray(ptch_2), seq_length)

n_patterns = [len(dataX1_p), len(dataX1_p_2)]
print ("Total Patterns: ", n_patterns)


# In[11]:


#reshape inputs to be [samples, time steps, features]
dataX1_o = np.reshape(dataX1_o, (dtlngth[0] - seq_length + 1, seq_length, 1))
dataX1_q = np.reshape(dataX1_q, (dtlngth[0] - seq_length + 1, seq_length, 1))
dataX1_p = np.reshape(dataX1_p, (dtlngth[0] - seq_length + 1, seq_length, 1))

dataX1_o_2 = np.reshape(dataX1_o_2, (dtlngth[1] - seq_length + 1, seq_length, 1))
dataX1_q_2 = np.reshape(dataX1_q_2, (dtlngth[1] - seq_length + 1, seq_length, 1))
dataX1_p_2 = np.reshape(dataX1_p_2, (dtlngth[1] - seq_length + 1, seq_length, 1))


# In[12]:


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


# In[13]:


# configure problem
n_steps_out = seq_length

n_units = 128
# In[17]:


train = load_model("/data/data1/users/el13102/weight/train.h5")
train_2 = load_model("/data/data1/users/el13102/weight/train_2.h5")
#infenc_tr = load_model("/data/data1/users/el13102/weight/127/ohe 40x2-20-540/infenc.h5")
#infdec_tr = load_model("/data/data1/users/el13102/weight/127/ohe 40x2-20-540/infdec.h5")

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
decoder_state_input_h = Input(shape=(n_units,), name='input_12')
decoder_state_input_c = Input(shape=(n_units,), name='input_13')
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
decoder_state_input_h_2 = Input(shape=(n_units,), name='input_14')
decoder_state_input_c_2 = Input(shape=(n_units,), name='input_15')
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



# In[18]:
f1 = open('/data/data1/users/el13102/weight/history1.pckl', 'rb')
history1 = pickle.load(f1)
f1.close()
f2 = open('/data/data1/users/el13102/weight/history2.pckl', 'rb')
history2 = pickle.load(f2)
f2.close()

print(history1)
print(history2)


# spot check some examples
ql_to_int = qldir(max_o, True)
int_to_ql = qldir(max_o, False)
for _ in range(0):
    i = randint(1, split_i[0])
    X1_o = np.reshape(to_categorical([ql_to_int[ql[0]] for ql in dataX1_o[i]], num_classes=n_features_o), (1, seq_length, n_features_o))
    X1_q = np.reshape(to_categorical([ql_to_int[ql[0]] for ql in dataX1_q[i]], num_classes=n_features_q), (1, seq_length, n_features_q))
    X1_p = np.reshape(to_categorical(dataX1_p[i], num_classes=n_features_p), (1, seq_length, n_features_p))
    target = predict_sequence(infenc, infdec, X1_o, X1_q, X1_p, n_steps_out, n_features_o, n_features_q, n_features_p)
    for j in range(seq_length):
        print('X_o=%s, y_o=%s, X_q=%s, y_q=%s, X_p=%s, y_p=%s' % (
            dataX1_o[i][j], int_to_ql[one_hot_decode([target[3*j]])[0]], 
            dataX1_q[i][j], int_to_ql[one_hot_decode([target[3*j+1]])[0]], 
            dataX1_p[i][j], one_hot_decode([target[3*j+2]])))
    print()


# In[19]:


# spot check some examples
ql_to_int = qldir(max_o, True)
int_to_ql = qldir(max_o, False)
for _ in range(0):
    i = randint(1, split_i[1])
    X1_o = np.reshape(to_categorical([ql_to_int[ql[0]] for ql in dataX1_o_v_2[i]], num_classes=n_features_o), (1, seq_length, n_features_o))
    X1_q = np.reshape(to_categorical([ql_to_int[ql[0]] for ql in dataX1_q_v_2[i]], num_classes=n_features_q), (1, seq_length, n_features_q))
    X1_p = np.reshape(to_categorical(dataX1_p_v_2[i], num_classes=n_features_p), (1, seq_length, n_features_p))
    target = predict_sequence(infenc, infdec_2, X1_o, X1_q, X1_p, n_steps_out, n_features_o, n_features_q, n_features_p)
    for j in range(seq_length):
        print('X_o=%s, y_o=%s, X_q=%s, y_q=%s, X_p=%s, y_p=%s' % (
            dataX1_o_v_2[i][j], int_to_ql[one_hot_decode([target[3*j]])[0]], 
            dataX1_q_v_2[i][j], int_to_ql[one_hot_decode([target[3*j+1]])[0]], 
            dataX1_p_v_2[i][j], one_hot_decode([target[3*j+2]])))
    print()


#value = train.evaluate(generatorex(dataX1_o, dataX1_q, dataX1_p, seq_length, n_features_o, n_features_q, n_features_p, max_o, max_q, batch_size=540), steps= (dtlngth[0]-split_i[0]) // 540, verbose=2)


#value_2 = train_2.evaluate(generatorex(dataX1_o_2, dataX1_q_2, dataX1_p_2, seq_length, n_features_o, n_features_q, n_features_p, max_o, max_q, batch_size=540), steps= (dtlngth[0]-split_i[0]) // 540, verbose=2)


pred_range = 10000
number_of_equal_elements = 0
X_inp = np.array([dataX1_o[:pred_range], dataX1_q[:pred_range], dataX1_p[:pred_range]])
for i in range(pred_range):
    X_pred_o = []
    X_pred_q = []
    X_pred_p = []
    X1_o = np.reshape(to_categorical([ql_to_int[ql[0]] for ql in dataX1_o[i]], num_classes=n_features_o), (1, seq_length, n_features_o))
    X1_q = np.reshape(to_categorical([ql_to_int[ql[0]] for ql in dataX1_q[i]], num_classes=n_features_q), (1, seq_length, n_features_q))
    X1_p = np.reshape(to_categorical(dataX1_p[i], num_classes=n_features_p), (1, seq_length, n_features_p))
    target = predict_sequence(infenc, infdec, X1_o, X1_q, X1_p, n_steps_out, n_features_o, n_features_q, n_features_p)
    for j in range(seq_length):
        X_pred_o = X_pred_o + [int_to_ql[one_hot_decode([target[3*j]])[0]]] 
        X_pred_q = X_pred_q + [int_to_ql[one_hot_decode([target[3*j+1]])[0]]] 
        X_pred_p = X_pred_p + [one_hot_decode([target[3*j+2]])]
    X_pred_o = np.reshape(X_pred_o, (1, seq_length, 1))
    X_pred_q = np.reshape(X_pred_q, (1, seq_length, 1))
    X_pred_p = np.reshape(X_pred_p, (1, seq_length, 1))
    #X_pred = np.array([X_pred_o, X_pred_q, X_pred_p])
    curpr_o = np.sum(X_inp[0][i]==X_pred_o)
    curpr_q = np.sum(X_inp[1][i]==X_pred_q)
    curpr_p = np.sum(X_inp[2][i]==X_pred_p)
    number_of_equal_elements += (curpr_o + curpr_q + curpr_p)


print("Evaluating", pred_range)
#print(X_inp, X_pred)
total_elements = 3*pred_range*seq_length
percentage = number_of_equal_elements/total_elements
print('number of identical elements: \t\t{}'.format(number_of_equal_elements))
print('total number of elements: \t\t{}'.format(total_elements))
print('percentage of identical elements: \t{:.2f}%'.format(percentage*100))
