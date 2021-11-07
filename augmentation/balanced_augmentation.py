#split all images
#for each image if counts of fert and unfert > 3x fish larva, augment and add counts to the total number
#when counts +-100 of fish larva,
#break
import glob
import cv2
import pandas as pd
import os 

from data_augmentation import *

#total_counts = [379,231,934]

def augmentation(file_path,newpath):
    imnames = glob.glob(file_path + '/*.jpg')
    for imname in imnames:
        im = cv2.imread(imname)
        labname = imname.replace('.jpg','.txt')
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

        fert_counts = len(labels[labels['class']==0])
        unfert_counts = len(labels[labels['class']==1])
        fl_counts = len(labels[labels['class']==2])

        if fert_counts > (1.5 * fl_counts) or unfert_counts >= (fl_counts):
            print(f'{imname} is valid, augmenting... counts are {fert_counts} fert, {unfert_counts} unfert and {fl_counts} FL ')
            ##augment

            mirror_labels , mirror_im = mirror_image(im,labels)

            filename = imname.split('/')[-1]

            slice_path = newpath + filename.replace('.jpg', '_mirrored.jpg')
            
            cv2.imwrite(slice_path,mirror_im)

            slice_labels_path = newpath + filename.replace('.jpg', '_mirrored.txt')

            mirror_labels.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')

            rotations = [90,180,270]
            image_ext = '.jpg'

            for angle in rotations:
                imname_path = imname[:-4]

                im = yoloRotatebbox(imname_path, image_ext, angle)

                bbox = im.rotateYolobbox()
                image = im.rotate_image()

                cv2.imwrite(imname+'_rotated_' + str(angle) + '.jpg', image)

                file_name = imname+'_rotated_' + str(angle) + '.txt'
                if os.path.exists(file_name):
                        os.remove(file_name)

                # to write the new rotated bboxes to file
                for i in bbox:
                        with open(file_name, 'a') as fout:
                                fout.writelines(
                                        ' '.join(map(str, cvFormattoYolo(i, im.rotate_image().shape[0], im.rotate_image().shape[1]))) + '\n')

                mirrored_path = slice_path[:-4]

                im_mirrored = yoloRotatebbox(mirrored_path, image_ext, angle)

                bbox_mirrored = im_mirrored.rotateYolobbox()
                image_mirrored = im_mirrored.rotate_image()

                cv2.imwrite(mirrored_path +'_rotated_' + str(angle) + '.jpg', image_mirrored)

                file_name2 = mirrored_path +'_rotated_' + str(angle) + '.txt'
                if os.path.exists(file_name2):
                        os.remove(file_name2)

                # to write the new rotated bboxes to file
                for i in bbox_mirrored:
                        with open(file_name2, 'a') as fout:
                                fout.writelines(
                                        ' '.join(map(str, cvFormattoYolo(i, im_mirrored.rotate_image().shape[0], im_mirrored.rotate_image().shape[1]))) + '\n')



            
if __name__ == '__main__':
    augmentation('processed_data/','processed_data/')
    print('============= AUGMENTATION FINISH =======================')
        






