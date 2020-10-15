import os
from distutils.dir_util import copy_tree
from shutil import copytree

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in dirs:
            return os.path.join(root, name)
			
#print('\t%s' % find('MatlabSrc', 'C:/Users/Papias/Desktop/thesis/MSongsDB'))

file1 = open('C:/Users/Papias/Desktop/thesis/Rock_Cleansed.txt', 'r') 
Lines = file1.readlines() 

dparth = 'C:/Users/Papias/Desktop/thesis/copy/Rock_Cleansed/'
count = 0
# Strips the newline character 
for line in Lines: 
    f = line.strip()
    print(f)
    copytree(find(f, 'C:/Users/Papias/Desktop/thesis/lpd_17/lpd_17_cleansed'), dparth + f)
    
    #print("Line{}: {}".format(count, line.strip())) 
	
	
