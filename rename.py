import os

### PARAMETERS ###
directory = "/Users/annealter/Documents/SU/Classes/10_CS230-DeepLearning/Project/dataset2/"
start_index = 0
end_index = 1319
start_filename = 8681

for i in range(start_index,end_index):
	old_num = str(i)
	new_num = str(i + start_filename)

	old_file = os.path.join(directory, old_num + ".npz")
	new_file = os.path.join(directory, new_num + ".npz")
	os.rename(old_file, new_file)