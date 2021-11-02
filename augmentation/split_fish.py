import os
import numpy as np
import shutil
import glob
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

os.makedirs('processed_data')
os.makedirs('false_data')

# get all image names
imnames = glob.glob('test_split/*.jpg')

# specify path for a new tiled dataset
newpath = 'processed_data/'
falsepath = 'false_data/'

# specify slice width=height
slice_size = 640

## todo, take in args 
## -- img size
## -- directory

# tile all images in a loop

for imname in imnames:
    im = cv2.imread(imname)

    ## very weird, when unmasked, you need to flip and mirror 
    ## but when it is masked you dont need to...  


    imr = im
    height = imr.shape[0]
    width = imr.shape[1]
    labname = imname.replace('.jpg', '.txt')
    labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

    dim_h = math.ceil(height/slice_size) 
    dim_w = math.ceil(width/slice_size) 
    slice_size_h = int(height/dim_h)
    slice_size_w = int(width/dim_w)

    
    # we need to rescale coordinates from 0-1 to real image height and width
    labels[['x1', 'w']] = labels[['x1', 'w']] * width
    labels[['y1', 'h']] = labels[['y1', 'h']] * height
    
    counter = 0
    print('Image:', imname)
    # create tiles and find intersection with bounding boxes for each tile

    for i in range(dim_h):
        for j in range(dim_w):
            slice_labels = []

            for k in range(len(labels)):
                new_x = (labels.iloc[k,:]['x1'] - j * slice_size_w) / slice_size_w
                new_y = (labels.iloc[k,:]['y1'] - i * slice_size_h) / slice_size_h
                new_width = (labels.iloc[k,:]['w'] )/slice_size_w
                new_height = (labels.iloc[k,:]['h'] )/slice_size_h

                if 0 <= new_x <= 1 and 0<=new_y<=1:
                    slice_labels.append([int(labels.iloc[k,:]['class']),new_x,new_y,new_width,new_height])


            slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])

            filename = imname.split('/')[-1]

            if len(slice_df) ==0:
                slice_labels_path = falsepath + filename.replace('.jpg', f'_{i}_{j}.txt')
                
                slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')

                sliced = imr[i*slice_size_h:(i+1)*slice_size_h, j*slice_size_w:(j+1)*slice_size_w]
                filename = imname.split('/')[-1]

                slice_path = falsepath + filename.replace('.jpg', f'_{i}_{j}.jpg')
                cv2.imwrite(slice_path,sliced)
            
            else:
                slice_labels_path = newpath + filename.replace('.jpg', f'_{i}_{j}.txt')

                slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')

                sliced = imr[i*slice_size_h:(i+1)*slice_size_h, j*slice_size_w:(j+1)*slice_size_w]

                filename = imname.split('/')[-1]

                slice_path = newpath + filename.replace('.jpg', f'_{i}_{j}.jpg')
        
                cv2.imwrite(slice_path,sliced)

            
            
                    

# # Creating Train / Val / Test folders (One time use)
root_dir = 'Fish_Data_Processed'
train_img = '/images/train'
val_img = '/images/val'
train_labels = '/labels/train'
val_labels = '/labels/val'
test_img = '/test_images'

os.makedirs(root_dir)
os.makedirs(root_dir + train_img)
os.makedirs(root_dir + val_img)
os.makedirs(root_dir + train_labels)
os.makedirs(root_dir + val_labels)
os.makedirs(root_dir + test_img)

src = 'processed_data'

allFileNames = glob.glob(src+'/*.jpg')
allFileNames = [x[:-4] for x in allFileNames]
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.8), int(len(allFileNames)*0.9)])



# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name+'.jpg', "Fish_Data_Processed/images/train/" )
    shutil.copy(name+'.txt', "Fish_Data_Processed/labels/train/" )

for name in val_FileNames:
    shutil.copy(name+'.jpg', "Fish_Data_Processed/images/val/" )
    shutil.copy(name+'.txt', "Fish_Data_Processed/labels/val/" )

for name in test_FileNames:
    shutil.copy(name+'.jpg', "Fish_Data_Processed/test_images/" )


file = open('Fish_Data_Processed/train.txt','w')
l = ['./images/train/'+name[15:]+'.jpg' + '\n' for name in train_FileNames]
for i in l:
    file.write(i)
file.close()

file = open('Fish_Data_Processed/val.txt','w')
l = ['./images/val/'+name[15:]+'.jpg'+ '\n' for name in val_FileNames]
for i in l:
    file.write(i)
file.close()

print('================== Done ================================= ')