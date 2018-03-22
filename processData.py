import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

### PARAMETERS ###
THRESHOLD = 0.5
m = 10000
n_h = 40
n_w = 40
n_C = 2
#NUM_FEATURES = 3*(41*41)
#NUM_ELEMENTS = 40*40
DIMENSION = 40
TRAIN_SIZE = 8000
DEV_SIZE = 1000
TEST_SIZE = 1000

### INITIALIZATION OF VECTORS ###
X  = np.zeros((m, n_h, n_w, n_C))				# [m, n_h, n_W, n_C]
Y  = np.zeros((m, n_h, n_w))				    # [m, n_h, n_W]
X_train = np.zeros((TRAIN_SIZE, n_h, n_w, n_C))
Y_train = np.zeros((TRAIN_SIZE, n_h, n_w))	
X_dev   = np.zeros((DEV_SIZE, n_h, n_w, n_C))	
Y_dev   = np.zeros((DEV_SIZE, n_h, n_w))	
X_test  = np.zeros((TEST_SIZE, n_h, n_w, n_C))	
Y_test  = np.zeros((TEST_SIZE, n_h, n_w))

x1 = np.zeros(((DIMENSION + 1)*(DIMENSION + 1), 1))
x2 = np.zeros(((DIMENSION + 1)*(DIMENSION + 1), 1))
x3 = np.zeros(((DIMENSION + 1)*(DIMENSION + 1), 1))

for i in range(1596,1597): #m  1595,1596  0,m)
	print i
	num = str(i)
	# load data from filepath
	filepath = "/Users/annealter/Documents/SU/Classes/10_CS230-DeepLearning/Project/dataset/" + num + ".npz"

	data = np.load(filepath)['arr_0']

	# INPUT DATA
	data_in_prev = data[4,:,:]
	data_in = data[5,:,:]
	X[i,:,:,0] = data_in                       # CURRENT LAYER
	X[i,:,:,1] = data_in - data_in_prev		   # GRADIENT BETWEEN CURRENT AND PREVIOUS LAYER

	# GROUND TRUTH DATA
	data_out = data[99,:,:]
	Y[i,:,:] = data_out > THRESHOLD

	# TEST TO SEE INPUTS + OUTPUTS
	# fig = plt.figure()
	# plt.imshow(dX[i,:,:], cmap="binary")
	# plt.show()

	fig = plt.figure()
	plt.imshow(data[5,:,:], cmap="binary")
	plt.show()

	fig = plt.figure()
	plt.imshow(data_in - data_in_prev, cmap="binary")
	plt.show()

	fig = plt.figure()
	plt.imshow(data[99,:,:]>0.5, cmap="binary")
	plt.show()

	# LOADS ON SYSTEM -- MAY USE THIS LATER IN PROJECT
	x1_in = np.load(filepath)['arr_1']				# fixed x
	x1_in = np.reshape(x1_in, (x1_in.shape[0],1))
	x1[x1_in-1,0]= 1								# one-hot representation of fixed nodes

	x2_in = np.load(filepath)['arr_2']				# fixed y
	x2_in = np.reshape(x2_in, (x2_in.shape[0],1))
	x2[x2_in-1,0]= 1								# one-hot representation of fixed nodes

	x3_in = np.load(filepath)['arr_3']				# load y
	x3_in = np.reshape(x3_in, (x3_in.shape[0],1))
	x3[x3_in-1,0]= 1								# one-hot representation of loads in y

# SHUFFLE DATA
# permutation = list(np.random.permutation(m))
# shuffled_X = X[permutation, :, :]
# shuffled_Y = Y[permutation, :, :]

# #SPLIT DATA INTO TEST,TRAIN,DEV
# X_train = shuffled_X[0:(TRAIN_SIZE),:,:,]
# X_dev   = shuffled_X[(TRAIN_SIZE):(TRAIN_SIZE+DEV_SIZE),:,:]
# X_test  = shuffled_X[(TRAIN_SIZE+DEV_SIZE):(TRAIN_SIZE+DEV_SIZE+TEST_SIZE),:,:]
# Y_train = shuffled_Y[0:(TRAIN_SIZE),:,:]
# Y_dev   = shuffled_Y[(TRAIN_SIZE):(TRAIN_SIZE+DEV_SIZE),:,:]
# Y_test  = shuffled_Y[(TRAIN_SIZE+DEV_SIZE):(TRAIN_SIZE+DEV_SIZE+TEST_SIZE),:,:]

# # SAVE DATA
# filepath = "/Users/annealter/Documents/SU/Classes/10_CS230-DeepLearning/Project/dataset_compiled/" + "X_train.npy"
# np.save(filepath, X_train)
# filepath = "/Users/annealter/Documents/SU/Classes/10_CS230-DeepLearning/Project/dataset_compiled/" + "Y_train.npy"
# np.save(filepath, Y_train)
# filepath = "/Users/annealter/Documents/SU/Classes/10_CS230-DeepLearning/Project/dataset_compiled/" + "X_dev.npy"
# np.save(filepath, X_dev)
# filepath = "/Users/annealter/Documents/SU/Classes/10_CS230-DeepLearning/Project/dataset_compiled/" + "Y_dev.npy"
# np.save(filepath, Y_dev)
# filepath = "/Users/annealter/Documents/SU/Classes/10_CS230-DeepLearning/Project/dataset_compiled/" + "X_test.npy"
# np.save(filepath, X_test)
# filepath = "/Users/annealter/Documents/SU/Classes/10_CS230-DeepLearning/Project/dataset_compiled/" + "Y_test.npy"
# np.save(filepath, Y_test)

print("done")


