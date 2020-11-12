import pypianoroll
import matplotlib
from matplotlib import pyplot
import os

root_dir = 'C:/Users/Papias/Desktop/thesis/copy/Rock_Cleansed'

for directory, subdirectories, files in os.walk(root_dir):
    for file in files:
        # print (os.path.abspath(os.path.join(directory, file)))
        npzf = pypianoroll.load(os.path.abspath(os.path.join(directory, file)))
        pypianoroll.write(npzf, 'C:/Users/Papias/Desktop/thesis/copy/midi/Rock_Cleansed/' + directory[-18:] + '.mid')
		
# fig, ax = pypianoroll.plot_multitrack(npzf, 'C:/Users/Papias/Desktop/thesis/copy/midi', mode='separate', track_label='name', preset='default', cmaps=None, xtick='auto', ytick='octave', xticklabel=True, yticklabel='auto', tick_loc=None, tick_direction='in', label='both', grid='both', grid_linestyle=':', grid_linewidth=0.5)
# pyplot.show()

