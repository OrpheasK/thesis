from music21 import converter, corpus, instrument, midi, note, chord, pitch, stream, tempo

def midi2keystrikes(filename,tracknum):
    """
    Reads a midifile (thanks to the package music21), returns a list
    of the keys hits:  [{'time':15, 'note':50} ,{... ]
    """
     
    mf = midi.MidiFile()
    mf.open(filename)
    mf.read() 
    mf.close()
    events = mf.tracks[tracknum].events
    result = []
    t=0
     
    for e in events:
         
        if e.isDeltaTime and (e.time is not None):
             
            t += e.time
             
        elif ( e.isNoteOn and ( e.pitch is not None) and
              (e.velocity != 0) and (e.pitch > 11)):
                    
            result.append( {'time':t, 'note':e.pitch} )
             
    if (len(result) == 0) and (tracknum <5):
        # if it didn't work, scan another track.
        return midi2keystrikes(filename,tracknum+1)
         
    return result
	
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
    
base_midi = open_midi("C:/Users/Papias/Desktop/thesis/copy/midi/midi.mid", True)


# base_midi.removeByClass('Piano')
# base_midi.write("midi", "C:/Users/Papias/Desktop/thesis/copy/midi/midirem.mid")

def list_instruments(midi):
    partStream = midi.parts.stream()
    print("List of instruments found on MIDI file:")
    for p in partStream:
        aux = p
        print (p.partName)
		
def move_instruments(midi):
    partStream = midi.parts.stream()
    s1 = stream.Stream()
    print("List of instruments found on MIDI file:")
    for p in partStream:
        if (p.partName != 'Piano'):
	        s1.append(p.partStream)
    return s1

# new_base = move_instruments(base_midi)	
# list_instruments(new_base)

# print (base_midi.parts[0].partName)

# print (midi2keystrikes("C:/Users/Papias/Desktop/thesis/copy/midi/midi.mid",0))




def extract_notes(midi_part):
    parent_element = []
    ret = []
    for nt in midi_part.flat.notes:        
        if isinstance(nt, note.Note):
            ret.append(max(0.0, nt.pitch.ps))
            parent_element.append(nt)
        elif isinstance(nt, chord.Chord):
            for pitch in nt.pitches:
                ret.append(max(0.0, pitch.ps))
                parent_element.append(nt)
    
    return ret, parent_element

def print_parts_countour(midi):

    
    # Drawing notes.
    for i in range(len(midi.parts)):
        top = midi.parts[i].flat.notes                  
        y, parent_element = extract_notes(top)
        if (len(y) < 1): continue
        print (i, y)
        print (i, parent_element)

# Focusing only on 6 first measures to make it easier to understand.
# print_parts_countour(base_midi.measures(0, 6))


# s = stream.Part(id='restyStream')
# s.append(note.Note('C#'))
# s.append(note.Rest(quarterLength=2.0))
# d=note.Note(pitch=61,quarterLength=2.0)
# d.quarterLength=2.0
# d.offset=3.0
# s.insert(30.5, d)
# s.append(note.Rest(quarterLength=1.0)) 

# s.write("text", "C:/Users/Papias/Desktop/thesis/copy/midi/mademidi.txt")
# print (d.offset)

#based on https://colab.research.google.com/github/cpmpercussion/creative-prediction/blob/master/notebooks/3-zeldic-musical-RNN.ipynb
stream_list = []
for element in base_midi.parts[0].flat:
	if isinstance(element, note.Note):
		stream_list.append([element.offset, element.quarterLength, element.pitch.midi])
	elif isinstance(element, chord.Chord):
		stream_list.append([element.offset, element.quarterLength, element.sortAscending().pitches[-1].midi])

		
# print (stream_list)
# print (stream_list[2][2])
s = stream.Part(id='restyStream')
mm1 = tempo.MetronomeMark(number=106) #works because track is 4/4

s.append(mm1)
i=0
for sl in stream_list:
	# if (i<10):
		# print (stream_list[i][0],stream_list[i][1],stream_list[i][2])
	d=note.Note(pitch=stream_list[i][2],quarterLength=stream_list[i][1])
	s.insert(stream_list[i][0], d)
	 
	i+=1
	
s.write("midi", "C:/Users/Papias/Desktop/thesis/copy/midi/mademidi.mid")
