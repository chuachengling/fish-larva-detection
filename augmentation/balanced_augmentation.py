#split all images
#for each image if counts of fert and unfert > 3x fish larva, augment and add counts to the total number
#when counts +-100 of fish larva,
#break
import glob
import cv2
import pandas as pd

from data_augmentation import *
from split_fist import *

total_counts = [379,231,934]

def augmentation(file_path,newpath,total_counts):
    imnames = glob.glob(file_path + '/*.jpg')
    for imname in imnames:
        im = cv2.imread(imname)
        labname = imname.replace('.jpg','.txt')
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
        counts = labels['class'].value_counts()
        fert_counts = counts[0]
        unfert_counts = counts[1]
        fl_counts = counts[2]

        if fert_counts > 2 * fl_counts or unfert_counts > 2 * fl_counts:
            ##augment
            mirror_labels , mirror_im, class_counts = mirror_image(im,labels)

            total_counts[0]  = total_counts[0] + class_counts[0] 
            total_counts[1]  = total_counts[1] + class_counts[0] 
            total_counts[2] = total_counts[2] + class_counts[0]

            if abs(total_counts[2] - total_counts[1]) < 100 and abs(total_counts[2] - total_counts[0]) < 100:
                break

        ## save img and label 

        filename = imname.split('/')[-1]

        slice_path = newpath + filename.replace('.jpg', '_mirrored.jpg')
        cv2.imwrite(slice_path,mirror_im)

        slice_labels_path = newpath + filename.replace('.jpg', '_mirrored.txt')

        mirror_labels.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')

        






