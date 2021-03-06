from music21 import converter, corpus, instrument, midi, note, chord, pitch, stream, tempo
import csv
from operator import itemgetter
import os

#https://www.kaggle.com/wfaria/midi-music-data-extraction-using-music21
def open_midi(midi_path, remove_tracks):
    # There is an one-line method to read MIDIs
    # but to remove the drums we need to manipulate some
    # low level MIDI events.
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    if (remove_tracks):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]          

    return midi.translate.midiFileToStream(mf)
    
# temp_midi_chords = open_midi("C:/Users/Papias/Desktop/thesis/copy/midi/midi.mid",True).chordify()
# base_midi = stream.Score()
# base_midi.insert(0, temp_midi_chords)

# a = stream.Stream()
# for v in base_midi.voices:
	# for n in v.notes:
		# a.insert(n.offset, n)

def textpr(b_midi, fname):		
	#based on https://colab.research.google.com/github/cpmpercussion/creative-prediction/blob/master/notebooks/3-zeldic-musical-RNN.ipynb
	stream_list = []

	for i in range(len(b_midi.parts)):
		for element in b_midi.parts[i].flat:
			if isinstance(element, note.Note):
				stream_list.append([float(element.offset), float(element.quarterLength), element.pitch.midi])
				# a.insert(element.offset, element)
			elif isinstance(element, chord.Chord):
				for nt in element.pitches:
					stream_list.append([float(element.offset), float(element.quarterLength), nt.midi])

	stream_list = sorted(stream_list, key=itemgetter(0))

	with open("C:/Users/Papias/Desktop/thesis/copy/midi/midi21txt/" + fname[:-4] + ".txt", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(stream_list)
	
for directory, subdirectories, files in os.walk('C:/Users/Papias/Desktop/thesis/copy/midi/Rock_Cleansed'):
    for file in files:
        base_midi = open_midi(os.path.abspath(os.path.join(directory, file)), True) #open and remove drums
        textpr(base_midi, file)