with open('C:/Users/Papias/Desktop/thesis/lastfm/id_list_trance_00s.txt', 'r') as file1:
    with open('C:/Users/Papias/Desktop/thesis/lastfm/id_list_trance_90s.txt', 'r') as file2:
        same = set(file1).intersection(file2)

same.discard('\n')

# with open('C:/Users/Papias/Desktop/thesis/lastfm/id_list_trance_00s.txt', 'w') as file_out:
    # for line in same:
        # file_out.write(line)	
		
for i, l in enumerate(same):
    pass
	
print(i + 1)