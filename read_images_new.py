#import pandas as pd
#import scipy.io
#from pandas import Series,DataFrame
#from PIL import Image
#from os import listdir
#from os.path import isfile, join
import numpy as np
import cv2
import os
#from tempfile import TemporaryFile

#image_names = []
path = "./small_1"
valid_images = [".jpg",".jpeg",".png",".tga",".bmp"]
num_images = len(os.listdir(path))
target_height = 212
target_width = 398
images = np.empty((num_images,target_height,target_width,3), dtype='uint8')
labels = np.zeros(shape=(num_images,))
n=0;
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    labels[n]=int(f[0])-1
    image = cv2.imread(os.path.join(path,f))
    image = image[40:-40,75:-75]
    images[n,:,:,:] = image
    n+=1
sel = np.random.permutation(num_images)
y = labels[sel]
x = images[sel]

ind = len(y)

x_train = x[:int(0.8*ind)]
y_train = y[:int(0.8*ind)]
# read validation data 
x_val = x[int(0.8*ind):int(0.9*ind)]
y_val = y[int(0.8*ind):int(0.9*ind)]
# read test data 
x_test = x[int(0.9*ind):]
y_test = y[int(0.9*ind):]
# checks
print('train shape:','\nData: ',x_train.shape,'\nlabels: ',y_train.shape)
print('val shape:','\nData: ',x_val.shape,'\nlabels: ',y_val.shape)
print('test shape:','\nData: ',x_test.shape,'\nlabels: ',y_test.shape)
print('saving data......')
# for compressive saving
np.savez_compressed('/home/ti1080/GP_Team#/drive-download-20170709T124015Z-001/Trian_data',y_train=y1,x_train=x_train)
np.savez_compressed('/home/ti1080/GP_Team#/drive-download-20170709T124015Z-001/Val_data',y_val=y2,x_val=x_val)
np.savez_compressed('/home/ti1080/GP_Team#/drive-download-20170709T124015Z-001/Test_data',y_test=y3,x_test=x_test)

## for loading
#loaded = np.load('/home/ti1080/GP_Team#/drive-download-20170709T124015Z-001/data.npz')
#y=loaded['y']
#x=loaded['x']
#for n in range(num_images):
    
    #cv2.imshow('image',image)
    #print(os.path.join(img_dir,image_names[n]))
    #image = cv2.resize(image,(target_width,target_height))
    #cv2.imshow('image',image)
    
