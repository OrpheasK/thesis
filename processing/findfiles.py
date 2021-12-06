import os
from distutils.dir_util import copy_tree
from shutil import copytree
import errno

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in dirs:
            return os.path.join(root, name)
			
#print('\t%s' % find('MatlabSrc', 'C:/Users/Papias/Desktop/thesis/MSongsDB'))

file1 = open('C:/Users/Papias/Desktop/thesis/lastfm/id_list_trance_90s_00s.txt', 'r') 
Lines = file1.readlines() 

dparth = 'C:/Users/Papias/Desktop/thesis/copy/lastfm/trance_cleansed/'
count = 0
# Strips the newline character 
for line in Lines: 
    f = line.strip()
    # print(f)
    try:
        copytree(find(f, 'C:/Users/Papias/Desktop/thesis/lpd_17/lpd_17_cleansed'), dparth + f)
    except FileExistsError:
        count += 1
        print('File already exists(', count, ')')
        continue # skip this url and proceed to the next
    
    #print("Line{}: {}".format(count, line.strip())) 
	
	
