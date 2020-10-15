import pypianoroll
import matplotlib
from matplotlib import pyplot

npzf = pypianoroll.load('C:/Users/Papias/Desktop/thesis/copy/Rock_Cleansed/TRABVVN12903CB6445/d809d7337042605a186eef7a81fa49f6.npz')
# fig, ax = pypianoroll.plot_multitrack(npzf, 'C:/Users/Papias/Desktop/thesis/copy/midi', mode='separate', track_label='name', preset='default', cmaps=None, xtick='auto', ytick='octave', xticklabel=True, yticklabel='auto', tick_loc=None, tick_direction='in', label='both', grid='both', grid_linestyle=':', grid_linewidth=0.5)
# pyplot.show()
pypianoroll.write(npzf, 'C:/Users/Papias/Desktop/thesis/copy/midi/altr2.mid')
