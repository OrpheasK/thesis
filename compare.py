with open('C:/Users/Papias/Desktop/thesis/id_list_Rock.txt', 'r') as file1:
    with open('C:/Users/Papias/Desktop/thesis/test2.txt', 'r') as file2:
        same = set(file1).intersection(file2)

same.discard('\n')

with open('C:/Users/Papias/Desktop/thesis/Rock_Cleansed.txt', 'w') as file_out:
    for line in same:
        file_out.write(line)	