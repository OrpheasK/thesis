# Import the os module, for the os.walk function
import os

def save_to_file(text):
    with open('C:/Users/Papias/Desktop/thesis/test2.txt', mode='a+', encoding='utf-8') as myfile:
        myfile.write(''.join(text))
        myfile.write('\n')
			
# Set the directory you want to start from
rootDir = 'C:/Users/Papias/Desktop/thesis/lpd_17/lpd_17_cleansed'

string2 = ".h5"
for dirName, subdirList, fileList in os.walk(rootDir):
    # print('Found directory: %s' % dirName)
    for fname in fileList:
        # if string2 in fname:
            # f = fname.replace(string2,"")
        subdirList = dirName.split(os.path.sep)[-1]
        # print('\t%s' % subdirList)
        # print('\t%s' % fname)
        save_to_file(subdirList)


