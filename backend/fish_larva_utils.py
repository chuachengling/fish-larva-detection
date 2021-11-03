import glob
import math
import cv2
import pandas as pd
import numpy as np

def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        #cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def cut_test_images(IMG_FILE,SAVE_PATH,SLICE_SIZE):
    im = IMG_FILE
    height = im.shape[0]
    width = im.shape[1]
    dim_h = math.ceil(height/SLICE_SIZE)
    dim_w = math.ceil(width/SLICE_SIZE)
    slice_size_h = int(height/dim_h)
    slice_size_w = int(width/dim_w)

    for i in range(dim_h):
        for j in range(dim_w):
            filename = 'image.jpg'
            sliced = im[i*slice_size_h:(i+1)*slice_size_h, j*slice_size_w:(j+1)*slice_size_w]
            slice_path = SAVE_PATH + filename.replace('.jpg', f'_{i}_{j}.jpg')
            cv2.imwrite(slice_path,sliced)


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def stitch_test_images(READ_PATH):
  imnames = glob.glob(READ_PATH + '*jpg')

  unique_images = set([imname[:-8] for imname in imnames]) 
  num_unique_images = len(unique_images) 
  h_len = int(max([imname[-7] for imname in imnames]))+1
  w_len = int(max([imname[-5] for imname in imnames]))+1

  for i in range(num_unique_images):
    image_name = list(unique_images)[i]
    image_list = []
    for imname in imnames:
      if imname[:-8] == image_name:
        image_list.append(imname)

    hori_imgs = [0] * h_len

    for k in range(w_len):
      for j in range(h_len):
        im = cv2.imread(image_name + f'_{j}_{k}.jpg')
        if not isinstance(hori_imgs[j],int):
          hori_imgs[j] = cv2.hconcat([hori_imgs[j],im ])
        else:
          hori_imgs[j] = im
    out = vconcat_resize_min(hori_imgs)
    file_name = image_name.split('/')[-1]


  return out


## show counts 
def generate_test_results(output):
  
  output = pd.DataFrame(output['class'].value_counts()).reset_index().rename(columns = {'index': 'class','class':'model_count'}).sort_values(by='class')
  output['class'] = output.apply(rename_class,axis=1)
  return output

def rename_class(row):
    if row['class'] == 0:
      return 'fertilized egg'
    if row['class'] == 1:
      return 'unfertilized egg'
    if row['class'] == 2:
      return 'fish larva'
    if row['class'] == 3:
      return 'unidentifiable object'
