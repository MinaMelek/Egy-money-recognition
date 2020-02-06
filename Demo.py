import scipy.io
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import os
import cv2
from math import floor, ceil, sqrt
#import shutil

### Useful functions
def Padding(image): # padding to constant sizes (292,548)
    p = 292
    q = 548
    s = image.shape
    m = s[0]
    n = s[1]
    b_top = floor((p-m)/2)
    b_bot = ceil((p-m)/2)
    b_left = floor((q-n)/2)
    b_right = ceil((q-n)/2)
    image = cv2.copyMakeBorder(image,b_top,b_bot,b_left,b_right,cv2.BORDER_CONSTANT,value=[0,0,0])
    return image

### The path to the selected images
path = ".\\6\\orig"
### Pre-definitions 
valid_x = [".jpg",".jpeg",".png",".tga",".bmp"] # valid image extensions
num_x = len(os.listdir(path))
# final sizes of the image
height = int(212/5)-2#40
width = int(390/5)+2#80
channels = 3
classes_num = 7
x = np.empty((num_x,height,width,3), dtype='float32')
# theoretical sizes for resizing stage_1
sized_hight=200
sized_width=430
sized=sized_hight*sized_width
n=0;
### Pre-processing
for f in os.listdir(path):
    ext = os.path.splitext(f)[1] # getting the file extension
    if ext.lower() not in valid_x:
        continue # pass if the file's extension not in the valid extensions list
    global j
    j =os.path.join(path,f) 
    image = cv2.imread(j) # reading images
    s=image.shape
    # rotat 90deg in case of hight>width
    if s[0]>s[1]:
        image = np.rot90(image, k=1)
        s=image.shape
    resized=s[0]*s[1]
    scale=resized/sized
    # resizing to scale (stage_1)
    image = cv2.resize(image,(floor(s[1]/sqrt(scale)),floor(s[0]/sqrt(scale))))
    # filtering
    image = cv2.medianBlur(image, 3)
    # padding to biggest sizes 
    image = Padding(image) # using the pre-defined "Padding" function upove 
    # cropping
    image = image[40:-40,79:-79]
    # resizing to final sizes(stage_2) 
    image = cv2.resize(image,(width,height))
    image = image/255
    x[n,:,:,:] = image
    n+=1
# read test data 
x_test = x

### The MODEL

# (construct the CNN model)
# define a sequential model
model = Sequential()
# adding a filters layer with 64 filter of size (3*3) and a relu activation function for each neuron
# and since this is the first layer we have to define the input shape in our case the input is a (42*78) picture with 1 channel
model.add(Conv2D(64, (3, 3), activation='relu' ,input_shape=(height,width,channels)))
# adding another filters layer with 128 filters of size (3*3) and a relu activation function for each neuron
model.add(Conv2D(128, (3, 3), activation='relu'))
# this is a sampling layer of a window size (2*2) which means out of each 4 neurons in the filters before we select one
model.add(MaxPooling2D(pool_size=(2, 2)))
# to prevent overfitting we add a dropout TO THE OUTPUT OF THE PREVIOUS LAYER ONLY which shuts down 25% of the connections during training selected at random each time
model.add(Dropout(0.25))
# another filters layer
model.add(Conv2D(256, (3, 3), activation='relu'))
# another filter layer
model.add(Conv2D(256, (3, 3), activation='relu'))
# another sampling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# a dropout layer
model.add(Dropout(0.25))
# another filters layer
model.add(Conv2D(256, (3, 3), activation='relu'))
# another filter layer
model.add(Conv2D(512, (3, 3), activation='relu'))
# another sampling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# a dropout layer
model.add(Dropout(0.25))
# flat the previous output into a single vector "prepare for a fuuly connected part of netwrok"
model.add(Flatten())
# a fully connected layer of 256 neuron and a relu activation
model.add(Dense(1024, activation='relu'))
# a dropout for previous layer 
model.add(Dropout(0.5))
# the last 7 neurons "softmax" layer for classification
model.add(Dense(7, activation='softmax'))


# (configure the learnig methode and hyper-parameters "optimizer")
# applying Gradient Descent methode "lr=learnig rate" 
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# apply previous confg. to our model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# saving weights of the model after every epoch in a given directory
filepath="../weightsCNN/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)


# using tensorbard to visualize the training and view learning curves to know when to stop and choose which epoch as the best
# while the code is running run the following command in your terminal while pointing to the script directory
tbCallBack = keras.callbacks.TensorBoard(log_dir='./', histogram_freq=0, write_graph=True)


# (training of the model)
# passing the training data and validation data along with how many examples to evaluate at a time "batch_size" and to loop over data how many times "epochs"
# and shuffle the data help for a faster convergence and better accuracies
model.load_weights('./weights-31-0.99.hdf5')
#model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=32, epochs=5, verbose=2, shuffle=True, callbacks=[checkpoint,tbCallBack])


# (evaluate the performance of the model on the test data) after loading the best weights into the model by using the following line
# (model.load_weights('my_best_weight.hdf5')
#performance=model.evaluate(x_test, y_test, batch_size=32, verbose=0)
#print("model accuracy is: ",performance[1]*100,'%')
#model_accuracy=performance[1]*100 # test accuracy
#x=input("To save your model and creat json file please press 's': ")
#if x=='s':
#    model.save_weights('model_save')
#    json_string = model.to_json()
#    text_file = open("model_json","w")
#    text_file.write(json_string)
#    text_file.close()
### Note: the next lines is preferred to be executed after loading the best weight file using "model.load_weights" function in previouse line
### otherwise it will be executed on the last result of training process

# Constructing confusion matrix and data used for fusion
labels1=model.predict(x_test,batch_size=32, verbose=0) # labels predicted from test data
# Definitions
y_test_pred1=[None]*len(labels1)

for i in range(len(labels1)):
    y_test_pred1[i]=list(labels1[i]).index(max(labels1[i])) # index of each sample's maximum value that represents the predicted class

### Printing_predictions
for mm in range(len(y_test_pred1)):
    if y_test_pred1[mm]==0:
        p="one pound - 1 LE."
    elif y_test_pred1[mm]==1:
        p="Five pounds - 5 LE"
    elif  y_test_pred1[mm]==2:
        p="Ten pounds - 10 LE"
    elif  y_test_pred1[mm]==3:
        p="Twenty pounds - 20 LE"
    elif  y_test_pred1[mm]==4:
        p="Fifty pounds - 50 LE"
    elif  y_test_pred1[mm]==5:
        p="One hundred pounds - 100 LE"
    elif  y_test_pred1[mm]==6:
        p="Two hundred pounds - 200 LE"
    else:
        p="Unexpected value"
            
    print("The currency predected value is: ",p)
#os.rename(j, ".\\image\\mm")
#shutil.move(j, ".\\image\\mm")
