# lstm autoencoder recreate sequence
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

# load the dataset
# csv of (time, note_duration, note, velocity) structure
dataset = loadtxt('mididata.csv', delimiter=',')

# define input sequence
sequence = dataset[:,:]

# reshape input into [samples, timesteps, features] ([songs, midi_events, selected_midi_features])
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 4))

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,3)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(sequence, sequence, epochs=300, verbose=0)
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')

# save recreation
yhat = model.predict(sequence, verbose=0)
numpy.savetxt("mididatarecr.csv", yhat, delimiter=",")

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))