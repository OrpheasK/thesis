from music21 import converter, corpus, instrument, midi, note, chord, pitch, stream, tempo
import csv 

with open("C:/Users/Papias/Desktop/thesis/copy/midi/midi21txt/Rock_Cleansed/3/TREEHIU128F425CA33.txt", 'r') as f: 
	reader = csv.reader(f)
	stream_list = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]

with open("C:/Users/Papias/Desktop/thesis/copy/midi/midi21txt/prod/TREEHIU128F425CA33inR_rtt.txt", 'r') as f: 
	reader = csv.reader(f)
	stream_list_inR = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]
	
s = stream.Part(id='restyStream')

mm1 = tempo.MetronomeMark(number=106) #works because track is 4/4
s.append(mm1)

inrlen = len(stream_list_inR)
i=0
for sl in stream_list:
	# if (i<10):
		# print (stream_list[i][0],stream_list[i][1],stream_list[i][2])
	if (i<inrlen):
		d=note.Note(pitch=stream_list[i][2],quarterLength=stream_list_inR[i][1])
		s.insert(stream_list_inR[i][0], d)
	 
	i+=1
	
s.write("midi", "C:/Users/Papias/Desktop/thesis/copy/midi/prod/TREEHIU128F425CA33inR_rtt.mid")