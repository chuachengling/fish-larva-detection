from fish_larva_utils import *

import math
import pandas as pd
import numpy as np
import torch

def inference(im,model,SLICE_SIZE=640):
    ## Get your max width and height
    height = im.shape[0]
    width = im.shape[1]
    dim_h = math.ceil(height/SLICE_SIZE)
    dim_w = math.ceil(width/SLICE_SIZE)
    slice_size_h = int(height/dim_h)
    slice_size_w = int(width/dim_w)

    labels=[]

    for i in range(dim_h):
        for j in range(dim_w):

            sliced = im[i*slice_size_h:(i+1)*slice_size_h, j*slice_size_w:(j+1)*slice_size_w]

            res = model(sliced)
            
            output = res.pandas().xywh[0]
            
            output['width'] = output['width']/width 
            output['height'] = output['height']/height

            output['xcenter'] = (j * slice_size_w + output['xcenter'] )/width
            output['ycenter'] = (i * slice_size_h + output['ycenter'] )/height

            labels.append(output)

    
    merged = pd.concat(labels)

    return merged 

def draw_bboxes(img,classes,colours,labels):

    im2 = np.ascontiguousarray(img, dtype=np.uint8)
    
    height ,width, _ = im2.shape

    for index,row in labels.iterrows():

        # Split string to float
        x, y, w, h = row['xcenter'],row['ycenter'],row['width'],row['height']

        x_center, y_center, w, h = float(x)*width, float(y)*height, float(w)*width, float(h)*height
        x1 = round(x_center-w/2)
        y1 = round(y_center-h/2)
        x2 = round(x_center+w/2)
        y2 = round(y_center+h/2) 

        class_idx = row['class']
        

        plot_one_box([x1,y1,x2,y2], im2, color=colours[class_idx], label=classes[class_idx], line_thickness=1)

    return im2

def transform_results(results):
    results_list = []
    for index,row in results.iterrows():
        bbox = row[['xcenter','ycenter','width','height']].tolist()
        predicted_class = row['class']
        class_name = row['name']
        confidence = row['confidence']
        output_dict = {
            "predicted_class":predicted_class,
            "class_name":class_name,
            "confidence":confidence,
            "bounding_box":bbox
            }
        results_list.append(output_dict)
    return results_list



