# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Model,load_model
# from tensorflow.keras.layers import UpSampling1D,Conv1D,Input,Flatten, LSTM, Dense,TimeDistributed,Lambda,Multiply,Add,Concatenate,Bidirectional,Conv2D,MaxPooling2D, Reshape,Average, MaxPooling1D,AveragePooling2D
# from tensorflow.keras.layers import AveragePooling1D
# from tensorflow.keras.optimizers import Adam
import numpy as np
import pypianoroll as pp


## Build the neural network
def gen_model(seq_len,num_classes,lr=10e-5,chroma=False,drumz = False,shallow=False,tree=False):

	def pooltree(inp):
		levels= list()
		levels.append(inp)
		for i in range(int(np.log2(seq_len))-1):
			levels.append(AveragePooling1D(2)(levels[-1]))
		return levels

	def shallow_block(inp):
		c = Conv1D(128,24,activation='relu',padding='same')(inp)
		c = MaxPooling1D(2)(c)
		return c

	def deep_block(inp):
		c = Conv1D(117,9,activation='relu',padding='same')(inp)
		c = Conv1D(117,9,activation='relu',padding='same')(c)
		c = Conv1D(128,9,activation='relu',padding='same')(c)
		c = MaxPooling1D(2)(c)
		return c


	inpu = Input((seq_len,128))
	
	curr_inp = inpu

	if tree:
		inps = pooltree(inpu)

	for i in range(int(np.log2(seq_len))-1):
		if shallow:
			o = shallow_block(curr_inp)
		else:
			o = deep_block(curr_inp)
		if tree:
			o = Concatenate(axis=-1)([o,inps[i+1]])
		curr_inp = o
	o = Flatten()(o)
	d = Dense(128,activation='relu')(o)
	d = Dense(num_classes,activation='sigmoid')(d)
	m = Model(inpu,d)
	opt = Adam(lr=lr)
	m.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acc',])
	m.summary()
	return m



## m is the neural network
## fname is the path to the midi file
## seq_len is the number of timesteps
def predict_song(fname, seq_len=1024):
	## Load MIDI to pianoroll
	mt = pp.Multitrack(fname)
	## Sum non percussive tracks
	tracks = [t for t in mt.tracks if not t.is_drum]
	x = sum([t.pianoroll for t in tracks])
	print(x.shape[0])

    ## Split pianoroll into sub-sequences of length seq_len
	hop = seq_len//2
	n = x.shape[0]//hop-1
	x_slice = np.zeros((n,seq_len,128))
	for i in range(n):
		## Overlapping sub-sequences
		x_slice[i]=x[i*hop:i*hop+seq_len]
		## Normalize
		x_slice[i] = x_slice[i]/(np.max(x_slice[i])+10e-7)
	## Predict on all subsequences
	# p = m.predict(x_slice)
	## Average the predictions
	# p = np.average(p,axis=0)
	return 



## The labels
topmagd=['Pop_Rock','Electronic','Country','RnB','Jazz','Latin','International','Rap','Vocal','New Age','Folk','Reggae','Blues']

## Build net and load weights

# Number of timesteps 
seq_len = 1024
## Build net
# m = gen_model(seq_len,len(topmagd),lr=10e-5,chroma=False,drumz = False,shallow=True,tree=False)
## Load weights
#mname = 'tree'+'False'+'shallow'+'True'+'topmagd_SEQ'+str(seq_len)+''+'0.0001'+'LR_B'
#m.load_weights(mname+'.h5')
# m.load_weights('/data/data1/users/el13102/weight/treeFalseshallowTruetopmagd_SEQ10240.0001LR_B.h5')

#midifile = '/media/eddie/Big_Drive/music-trees/datasets/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]/testROCK/'
#midifile = midifile + 'TheBeatles.Here_Comes_the_Sun.mid'
midifile = 'C:/Users/Papias/Desktop/thesis/copy/midi/Rock_Cleansed/1/TRABQUM12903C9A618.mid'

pred = predict_song(midifile)

# for i in range(pred.shape[0]):
	# print(topmagd[i]+' : '+str(pred[i]))
