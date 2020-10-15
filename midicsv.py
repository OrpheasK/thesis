import py_midicsv as pm
import csv
from numpy import genfromtxt
import numpy as np
from io import StringIO
import pandas as pd
# Load the MIDI file and parse it into CSV format
csv_string = pm.midi_to_csv("C:/Users/Papias/Desktop/thesis/copy/midi/midi.mid")

# Parse the CSV output of the previous command back into a MIDI file
midi_object = pm.csv_to_midi(csv_string)

#Save the parsed MIDI file to disk
# with open("C:/Users/Papias/Desktop/thesis/copy/midi/csvmidi.mid", "wb") as output_file:
    # midi_writer = pm.FileWriter(output_file)
    # midi_writer.write(midi_object)
	
with open('C:/Users/Papias/Desktop/thesis/copy/midi/test6.txt','w') as file:
    for line in csv_string:
        file.write(line)
        # file.write('\n')
	
# my_cols = ["A", "B", "C", "D", "E", "F", "G"]
# my_df=pd.read_csv('C:/Users/Papias/Desktop/thesis/copy/midi/test6.csv', names=my_cols, engine='python')
# print(my_df.to_numpy()[8][4])

# scores = []
# dates = []
# numbers = []
# i = 0
# with open('C:/Users/Papias/Desktop/thesis/copy/midi/test6.csv') as csvDataFile:
    # csvReader = csv.reader(csvDataFile)
    # for row in csvReader:
        # scores.append([])
        # scores[i].append(row[1])
        # scores[i].append(row[2])
        # scores[i].append(row[0])
        # i=i+1
        # dates.append(row[2])
        		
# print(scores[6][1])

# my_data = genfromtxt('C:/Users/Papias/Desktop/thesis/copy/midi/test6.csv', delimiter=',')

# reader = csv.reader(reader = csv.reader(scsv.split('\n'), delimiter=',')
# for row in reader:
    # print('\t'.join(row)).split('\n'), delimiter=',')
# for row in reader:
    # print('\t'.join(row))