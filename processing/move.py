import os
import shutil

def filtered_copy(src_dir, dest_dir, filter):
    print ('Copying files named ' + filter + ' in ' + src_dir + ' to ' + dest_dir)
    ignore_func = lambda d, files: [f for f in files if isfile(join(d, f)) and f != filter]
    # if os.path.exists(dest_dir):
        # print ('deleting existing data')
        # shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir, ignore=ignore_func)
	
	
filtered_copy('C:/Users/Papias/Desktop/thesis/lpd_17/lpd_17_cleansed/C/C/D', 'C:/Users/Papias/Desktop/thesis/copy', 'e227daa24a478b8e6211f4915d62809d.npz')