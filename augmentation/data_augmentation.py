import os
import numpy as np
import shutil
import glob
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import argparse

#os.makedirs('processed_data')

## todo, take in args 
## -- img size
## -- directory

# tile all images in a loop



def mirror_all(imnames,newpath='augment_images/'):

    for imname in imnames:
        im = cv2.imread(imname)

        height = im.shape[0]
        width = im.shape[1]
        labname = imname.replace('.jpg', '.txt')
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

        labels['x1'] = labels['x1'].apply(lambda x:1 - x)
        #labels['y1'] = labels['y1'].apply(lambda x:1 - x)

        output = cv2.flip(im,1)
        filename = imname.split('/')[-1]

        slice_path = newpath + filename.replace('.jpg', '_mirrored.jpg')
        cv2.imwrite(slice_path,output)

        slice_labels_path = newpath + filename.replace('.jpg', '_mirrored.txt')

        labels.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')

def mirror_image(img,labels):
    height = rotate_image.shape[0]
    width = img.shape[1]

    labels['x1'] = labels['x1'].apply(lambda x:1 - x)
    output = cv2.flip(img,1)

    class_counts = labels['class'].value_counts()

    return labels, output, class_counts



class yoloRotatebbox:
    def __init__(self, filename, image_ext, angle):
        assert os.path.isfile(filename + image_ext)
        assert os.path.isfile(filename + '.txt')

        self.filename = filename
        self.image_ext = image_ext
        self.angle = angle

        # Read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext, 1)

        rotation_angle = self.angle * np.pi / 180
        self.rot_matrix = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])

    def rotateYolobbox(self):

        new_height, new_width = self.rotate_image().shape[:2]

        f = open(self.filename + '.txt', 'r')

        f1 = f.readlines()

        new_bbox = []

        H, W = self.image.shape[:2]

        for x in f1:
            bbox = x.strip('\n').split(' ')
            if len(bbox) > 1:
                (center_x, center_y, bbox_width, bbox_height) = yoloFormattocv(float(bbox[1]), float(bbox[2]),
                                                                               float(bbox[3]), float(bbox[4]), H, W)

                upper_left_corner_shift = (center_x - W / 2, -H / 2 + center_y)
                upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y)
                lower_left_corner_shift = (center_x - W / 2, -H / 2 + bbox_height)
                lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)

                new_lower_right_corner = [-1, -1]
                new_upper_left_corner = []

                for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift,
                          lower_right_corner_shift):
                    new_coords = np.matmul(self.rot_matrix, np.array((i[0], -i[1])))
                    x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                    if new_lower_right_corner[0] < x_prime:
                        new_lower_right_corner[0] = x_prime
                    if new_lower_right_corner[1] < y_prime:
                        new_lower_right_corner[1] = y_prime

                    if len(new_upper_left_corner) > 0:
                        if new_upper_left_corner[0] > x_prime:
                            new_upper_left_corner[0] = x_prime
                        if new_upper_left_corner[1] > y_prime:
                            new_upper_left_corner[1] = y_prime
                    else:
                        new_upper_left_corner.append(x_prime)
                        new_upper_left_corner.append(y_prime)
                #             print(x_prime, y_prime)

                new_bbox.append([bbox[0], new_upper_left_corner[0], new_upper_left_corner[1],
                                 new_lower_right_corner[0], new_lower_right_corner[1]])

        return new_bbox

    def rotate_image(self):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        height, width = self.image.shape[:2]  # image shape has 3 dimensions
        image_center = (width / 2,
                        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origin) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(self.image, rotation_mat, (bound_w, bound_h))
        return rotated_mat

    
    

   
#convert from Yolo_mark to opencv format
def yoloFormattocv(x1, y1, x2, y2, H, W):
    bbox_width = x2 * W
    bbox_height = y2 * H
    center_x = x1 * W
    center_y = y1 * H

    voc = []

    voc.append(center_x - (bbox_width / 2))
    voc.append(center_y - (bbox_height / 2))
    voc.append(center_x + (bbox_width / 2))
    voc.append(center_y + (bbox_height / 2))

    return [int(v) for v in voc]

# convert from opencv format to yolo format
# H,W is the image height and width
def cvFormattoYolo(corner, H, W):
    bbox_W = corner[3] - corner[1]
    bbox_H = corner[4] - corner[2]

    center_bbox_x = (corner[1] + corner[3]) / 2
    center_bbox_y = (corner[2] + corner[4]) / 2

    return corner[0], round(center_bbox_x / W, 6), round(center_bbox_y / H, 6), round(bbox_W / W, 6), round(bbox_H / H,
                                                                                                            6)




def rotate_all(imnames,rotate_angles = [90,180,270],newpath = 'augment_images/'):
    image_ext = '.jpg'
    for imname in imnames:
        imname = imname[:-4]
        for angle in rotate_angles:
                print(f'Rotating {imname} by {angle}')
                
                im = yoloRotatebbox(imname, image_ext, angle)

                bbox = im.rotateYolobbox()
                image = im.rotate_image()

                # to write rotateed image to disk
                cv2.imwrite(imname+'_rotated_' + str(angle) + '.jpg', image)

                file_name = imname+'_rotated_' + str(angle) + '.txt'
                if os.path.exists(file_name):
                        os.remove(file_name)

                # to write the new rotated bboxes to file
                for i in bbox:
                        with open(file_name, 'a') as fout:
                                fout.writelines(
                                        ' '.join(map(str, cvFormattoYolo(i, im.rotate_image().shape[0], im.rotate_image().shape[1]))) + '\n')

def rotate_image(img,rotate_angle):
    pass



def scale_up(imnames,scale_value = 1.25):
    pass


def scale_down(imname,scale_value = 0.75):
    pass

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("-source", default="augment_images", help = "Source folder with images and labels needed to be augmented")
        #parser.add_argument("-target", default="augment_images", help = "Target folder for a new sliced dataset")
        #parser.add_argument("-ext", default=".jpg", help = "Image extension in a dataset. Default: .JPG")
        #parser.add_argument("-falsefolder", default=None, help = "Folder for tiles without bounding boxes")
        #parser.add_argument("-size", type=int, default=416, help = "Size of a tile. Dafault: 416")
        #parser.add_argument("-ratio", type=float, default=0.8, help = "Train/test split ratio. Dafault: 0.8")
        args = parser.parse_args()

        # get all image names
        imnames = glob.glob(f'{args.source}/*.jpg')

        # specify path for a new tiled dataset
        newpath = 'processed_data/'


        mirror_all(imnames,f'{args.source}/')

        print('Mirroring Done!')

        imnames = glob.glob('augment_images/*.jpg')

        rotate_all(imnames,newpath = f'{args.source}/')

        print('Rotations Done!')
        imnames = glob.glob('augment_images/*.jpg')   

    ## augment mirror all files inside 

    ## augment rotate

    ## scale up * 1.25

    ## scale down * 0.75

