import os, shutil
from distutils.dir_util import copy_tree
from shutil import copytree
import errno

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
			
#print('\t%s' % find('MatlabSrc', 'C:/Users/Papias/Desktop/thesis/MSongsDB'))

file1 = open('C:/Users/Papias/Desktop/thesis/copy/midi/Jazzset.txt', 'r') 
Lines = file1.readlines() 

dparth = 'C:/Users/Papias/Desktop/thesis/copy/midi/lastfm/forprod/'
count = 0
# Strips the newline character 
for line in Lines: 
    f = line.strip()
    # print(f)
    try:
        shutil.copy(find(f[:-4]+'.mid', 'C:/Users/Papias/Desktop/thesis/copy/midi/lastfm/jazz_cleansed'), dparth + f[:-4]+'.mid')
    except FileExistsError:
        count += 1
        print('File already exists(', count, ')')
        continue # skip this url and proceed to the next