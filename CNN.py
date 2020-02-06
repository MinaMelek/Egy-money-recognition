import scipy.io
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import cv2
import os

#image_names = []
path = "./small_1"
valid_images = [".jpg",".jpeg",".png",".tga",".bmp"]
num_images = len(os.listdir(path))
height = 292
width = 548
channels = 3
classes_num = 7
images2 = np.empty((num_images,height,width,channels), dtype='uint8')
labels2 = np.zeros(shape=(num_images,))
n=0;
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    labels2[n]=int(f[0])-1
    image = cv2.imread(os.path.join(path,f))
    images2[n,:,:,:] = image
    n+=1
sel = np.random.permutation(num_images)
y = labels2[sel]
x = images2[sel]
## for loading >>>> don't forget to comment the saving command below
#loaded = np.load('/home/ti1080/GP_Team#/drive-download-20170709T124015Z-001/Train_data.npz')
#y1=loaded['y_train']
#y_train = keras.utils.to_categorical(y1,num_classes=classes_num)
#x_train=loaded['x_train']
#loaded = np.load('/home/ti1080/GP_Team#/drive-download-20170709T124015Z-001/Val_data.npz')
#y2=loaded['y_train']
#y_train = keras.utils.to_categorical(y1,num_classes=classes_num)
#x_train=loaded['x_train']

### NOTE: the next code is trained using Training and Validaton Data on server 
### and tested with the Test Data on CPU based PC using packages included in "Anaconda3-4.2.0-Windows-x86_64" plus TensorFlow and Keras packages

# Reading Data 
#mat = scipy.io.loadmat('classifier_data.mat')

ind = len(y)

x_train = x[:int(0.8*ind)]
y_train = keras.utils.to_categorical(y[:int(0.8*ind)],num_classes=classes_num)
# read validation data 
x_val = x[int(0.8*ind):int(0.9*ind)]
y_val = keras.utils.to_categorical(y[int(0.8*ind):int(0.9*ind)],num_classes=classes_num)
# read test data 
x_test = x[int(0.9*ind):]
y_test = keras.utils.to_categorical(y[int(0.9*ind):],num_classes=classes_num)
# The next list contains multible lists each represents the number of filters used in each convolutional layer respectively
# For example layers[x] = [N1,N2,N3,N4,N5] explained in more detailes in APPENDIX III
layers=[[32,64,64,128,256],[64,64,64,64,64],[128,128,256,256,512],[32,32,64,64,128]
,[64,64,128,128,256],[64,128,128,256,512],[32,64,128,256,0],[64,128,256,256,512],[64,128,128,256,256]]
l=7 # This variable takes numbers from 0 to 8 to refers to the index of the previouse list more detailes in APPENDIX III
print("at L",l)
# Using sequential model as the CNN structure model
model = Sequential()
# Adding a Convolution filters layer with 'N1' filter of size (3*3) and a relu activation function for each neuron
# Since this is the first layer we have to define the input shape in our case the input is a (20*146) picture with 1 channel
model.add(Conv2D(layers[l][0], (5, 5), activation='relu',input_shape=(height,width,channels)))
# Adding another Convolution filters layer with 'N2' filters of size (3*3) and a relu activation function for each neuron
model.add(Conv2D(layers[l][1], (5, 5), activation='relu'))
# This is a subsampling layer of a window size (2*2) which means out of each 4 neurons in the filters before we select one
model.add(MaxPooling2D(pool_size=(2, 2)))
# Preventing overfitting by adding a dropout TO THE OUTPUT OF THE PREVIOUS LAYER ONLY which shuts down 25% of the connections during training selected at random each time
model.add(Dropout(0.25))
# Adding another Convolution filters layer with 'N3' filters of size (3*3) and a relu activation function for each neuron
model.add(Conv2D(layers[l][2], (5, 5), activation='relu'))
# Adding another Convolution filters layer with 'N4' filters of size (3*3) and a relu activation function for each neuron
model.add(Conv2D(layers[l][3], (5, 5), activation='relu'))
# this is a sampling layer of a window size (2*2) which means out of each 4 neurons in the filters before we select one
model.add(MaxPooling2D(pool_size=(2, 2)))
# a dropout layer
model.add(Dropout(0.25))
# Adding another Convolution filters layer with 'N3' filters of size (3*3) and a relu activation function for each neuron
model.add(Conv2D(layers[l][2], (5, 5), activation='relu'))
# Adding another Convolution filters layer with 'N4' filters of size (3*3) and a relu activation function for each neuron
model.add(Conv2D(layers[l][3], (5, 5), activation='relu'))
# this is a sampling layer of a window size (2*2) which means out of each 4 neurons in the filters before we select one
model.add(MaxPooling2D(pool_size=(2, 2)))
# a dropout layer
model.add(Dropout(0.25))
# Adding another Convolution filters layer with 'N3' filters of size (3*3) and a relu activation function for each neuron
model.add(Conv2D(layers[l][4], (5, 5), activation='relu'))
# Adding another Convolution filters layer with 'N4' filters of size (3*3) and a relu activation function for each neuron
model.add(Conv2D(layers[l][4], (5, 5), activation='relu'))
# this is a sampling layer of a window size (2*2) which means out of each 4 neurons in the filters before we select one
model.add(MaxPooling2D(pool_size=(2, 2)))
# a dropout layer
model.add(Dropout(0.25))
# Adding another Convolution filters layer with 'N3' filters of size (3*3) and a relu activation function for each neuron
model.add(Conv2D(layers[l][4], (5, 5), activation='relu'))
# Adding another Convolution filters layer with 'N4' filters of size (3*3) and a relu activation function for each neuron
model.add(Conv2D(layers[l][4], (5, 5), activation='relu'))
# this is a sampling layer of a window size (2*2) which means out of each 4 neurons in the filters before we select one
model.add(MaxPooling2D(pool_size=(2, 2)))
# a dropout layer
model.add(Dropout(0.25))
# Adding another Convolution filters layer with 'N5' filters of size (3*3) and a relu activation function for each neuron
### Note: this line should be uncommented in case if 'l' doesn't equal to  6 or 7
# model.add(Conv2D(layers[l][4], (2, 2), activation='relu'))

# Flat the previous output into a single vector "prepare for a fully connected part of netwrok"
model.add(Flatten())
# a fully-connected layer of 256 neuron and a relu activation
M = 512 # Fully-connected hidden layer nodes 
# Had values of 256, 1024, 2048
model.add(Dense(M, activation='relu'))
# a dropout for previous layer which shuts down 50% of the connections
model.add(Dropout(0.5))
# the last 7 neurons "softmax" layer for classification
model.add(Dense(classes_num, activation='softmax'))

# Configuring the learnig methode and hyper-parameters "optimizer"
# applying Gradient Descent methode "lr=learnig rate" 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# apply previous confg. to model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# saving weights of the model after every epoch in a given directory
# The name of the weights file could be recognized from a name containing epoch number then used 'Lx' as explained then accuracy of validation data
filepath="./weightsCNN/weights-{epoch:02d}-L"+str(l)+"-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)

# using tensorbard to visualize the training and view learning curves to know when to stop and choose which epoch as the best
tbCallBack = keras.callbacks.TensorBoard(log_dir='./', histogram_freq=0, write_graph=True)

# Training of the model
# passing the training data and validation data along with how many examples to evaluate at a time "batch_size" and to loop over data how many times "epochs"
# and shuffle the data help for a faster convergence and better accuracies
model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=32, epochs=3, verbose=2, shuffle=True, callbacks=[checkpoint,tbCallBack])
### Note: during training the previous line is active while the next line is commented 
### But during testing the previous line is commented while the next line is active 
# """model.load_weights('weights-35-7-0.94.hdf5')""" # Evaluating the performance of the model on the test data, after loading the best weights into the model

performance=model.evaluate(x_test, y_test, batch_size=32, verbose=0)
print("model accuracy is: ",performance[1]*100,'%')
model_accuracy=performance[1]*100 # test accuracy

### Note: the next lines is preferred to be executed after loading the best weight file using "model.load_weights" function in previouse line
### otherwise it will be executed on the last result of training process

# Constructing confusion matrix and data used for fusion
labels1=model.predict(x_test,batch_size=32, verbose=0) # labels predicted from test data
# Definitions
#y_test_pred1=[None]*len(labels1)
#y_test_orig=[None]*len(y_test)
#conf_mat=np.zeros(shape=(classes_num,classes_num))
#
#for i in range(len(y_test)):
#    y_test_pred1[i]=list(labels1[i]).index(max(labels1[i])) # index of each sample's maximum value that represents the predicted class
#    y_test_orig[i]=list(y_test[i]).index(max(y_test[i])) # the same as the previous line for original test labels
#    conf_mat[y_test_orig[i]][y_test_pred1[i]]+=1 # adding number of matches and mismatches to confusion matrix
#
#classes=[None]*classes_num
#Classes_acc=[None]*classes_num
#for k in range(classes_num):
#	# evaluating the accuracy of each class
#    indices = [i for i, j in enumerate(y_test_orig) if j == k] # indices of class k
#    classes[k]=list( x_test[i] for i in indices ) 
#    A = np.array(classes[k]) # test samples that belongs to class k
#    B = list( y_test[i] for i in indices ) 
#    B = np.array(B) # test labels that belongs to class k
#    performance=model.evaluate(A, B, batch_size=32, verbose=0)
#    print("Class ",k+1, " accuracy is: ",performance[1]*100,'%') # printing accuracy of class k
#    Classes_acc[k]=performance[1]*100
#	# representing accuracies in confusion matrix
#    conf_mat[k][:]/=len(indices)
#conf_mat*=100    
## Saving constructed data in a .mat file to ease its reading in MATLAB for confusion
#scipy.io.savemat('CNN_variables_originalTest.mat',{'cnnConfusionMatrix':conf_mat,'predictedLabels':y_test_pred1,'classes':classes
#                                                   ,'originalLabels':y_test_orig,'model_accuracy':model_accuracy,'Classes_accuracy':Classes_acc})

print('saving data matrix....')
print('if you want to save the data sets press [s]')
inp = input()
# for compressive saving
if inp == 's':
    np.savez_compressed('/home/ti1080/GP_Team#/drive-download-20170709T124015Z-001/Trian_data',y_train=y[:int(0.8*ind)],x_train=x_train)
    np.savez_compressed('/home/ti1080/GP_Team#/drive-download-20170709T124015Z-001/Val_data',y_val=y[int(0.8*ind):int(0.9*ind)],x_val=x_val)
    np.savez_compressed('/home/ti1080/GP_Team#/drive-download-20170709T124015Z-001/Test_data',y_test=y[int(0.9*ind):],x_test=x_test)
