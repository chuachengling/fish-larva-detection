import os 
import shutil 
import numpy as np
import glob  

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