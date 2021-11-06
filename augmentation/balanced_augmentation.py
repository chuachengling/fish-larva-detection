import glob
import cv2
import pandas as pd

from data_augmentation import *
from split_fist import *

def augmentation(file_path,newpath):
    imnames = glob.glob(file_path + '*.jpg')
    for imname in imnames:
        ##zero_unmirror refers to the 'original' img, rotated 0 degrees and unmirrored
        zero_unmirror_im = cv2.imread(imname)
        labname = imname.replace('.jpg','.txt')
        zero_unmirror_labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
        
        ##get the class counts for this image
        counts = zero_unmirror_labels['class'].value_counts()
        fert_counts=0
        unfert_counts = 0
        fl_counts = 0
        for i in (counts.index):
          if i==0:
            fert_counts = counts[0]
          if i==1:
            unfert_counts = counts[1]
          if i==2:
            fl_counts = counts[2]
            
        ##if the image meets passes the threshold ratio, augment 6 times
        if fert_counts > 2 * fl_counts or unfert_counts > 1.2 * fl_counts:
            
            ##augment: 0 degrees, mirrored
            zero_mirror_labels , zero_mirror_im = mirror_image(zero_unmirror_im, zero_unmirror_labels)
            
            ##augment: 90 degrees, unmirrored
            ninety_rotatedinstance = yoloRotatebbox(imname[:-4], '.jpg', 90)
            ninety_unmirror_im = ninety_rotatedinstance.rotate_image()
            ninety_unmirror_labels = ninety_rotatedinstance.rotateYolobbox()

            ##augment: 90 degrees, mirrored
            ninety_mirror_labels, ninety_mirror_im = mirror_image(ninety_unmirror_im, ninety_unmirror_labels)
            
            ##augment: 180 degrees, unmirrored
            oneeighty_rotatedinstance = yoloRotatebbox(imname[:-4], '.jpg', 180)
            oneeighty_unmirror_im = oneeighty_rotatedinstance.rotate_image()
            oneeighty_unmirror_labels = oneeighty_rotatedinstance.rotateYolobbox()

            ##augment: 180 degrees, mirrored
            oneeighty_mirror_labels, oneeighty_mirror_im = mirror_image(oneeighty_unmirror_im, oneeighty_unmirror_labels)
            
            ##augment: 270 degrees, unmirrored
            twoseventy_rotatedinstance = yoloRotatebbox(imname[:-4], '.jpg', 270)
            twoseventy_unmirror_im = twoseventy_rotatedinstance.rotate_image()
            twoseventy_unmirror_labels = twoseventy_rotatedinstance.rotateYolobbox()
            
            #prepare to save images and labels
            filename = imname.split('/')[-1]

            #save all the images
            cv2.imwrite(newpath + filename.replace('.jpg', '_0_mirrored.jpg'), zero_mirror_im)
            cv2.imwrite(newpath + filename.replace('.jpg', '_90_unmirrored.jpg'), ninety_unmirror_im)
            cv2.imwrite(newpath + filename.replace('.jpg', '_90_mirrored.jpg'), ninety_mirror_im)
            cv2.imwrite(newpath + filename.replace('.jpg', '_180_unmirrored.jpg'), oneeighty_unmirror_im)
            cv2.imwrite(newpath + filename.replace('.jpg', '_180_mirrored.jpg'), oneeighty_mirror_im)
            cv2.imwrite(newpath + filename.replace('.jpg', '_270_unmirrored.jpg'), twoseventy_unmirror_im)

            #save all labels
            zero_mirror_labels.to_csv(newpath + filename.replace('.jpg', '_0_mirrored.txt'), sep=' ', index=False, header=False, float_format='%.6f')
            ninety_unmirror_labels.to_csv(newpath + filename.replace('.jpg', '_90_unmirrored.txt'), sep=' ', index=False, header=False, float_format='%.6f')
            ninety_mirror_labels.to_csv(newpath + filename.replace('.jpg', '_90_mirrored.txt'), sep=' ', index=False, header=False, float_format='%.6f')
            oneeighty_unmirror_labels.to_csv(newpath + filename.replace('.jpg', '_180_unmirrored.txt'), sep=' ', index=False, header=False, float_format='%.6f')
            oneeighty_mirror_labels.to_csv(newpath + filename.replace('.jpg', '_180_mirrored.txt'), sep=' ', index=False, header=False, float_format='%.6f')
            twoseventy_unmirror_labels.to_csv(newpath + filename.replace('.jpg', '_270_unmirrored.txt'), sep=' ', index=False, header=False, float_format='%.6f')






