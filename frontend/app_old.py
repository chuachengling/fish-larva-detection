import streamlit as st 
from utils_fl import *
import glob
import math
import cv2
import pandas as pd
import subprocess
import os
import shutil
from PIL import Image
import numpy as np
import tempfile
from io import BytesIO
import torch
import random

from yolov5.utils.plots import Annotator

st.title('Welcome to DSA4266 Group 1 Fish Larva Detection Application')

file = st.file_uploader('Please upload an image',type = ['png','jpg','jpeg'])

if file is not None:
    ##mkdir
    file_paths = ['test_image_split/','test_data/']
    for file_path in file_paths:
        try:
            os.mkdir(file_path)
        except Exception as e:
            pass

    #IMAGE_PATH = './data/'
    SLICE_SIZE = 640
    SAVE_PATH = 'test_image_split/'
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)[:,:,::-1] 

    st.image(img)

    cut_test_images(img,SAVE_PATH,SLICE_SIZE)

    ## RUN INFERENCE 

    weights = st.selectbox('Please choose the weights you want', ['yolov5'])

    model = torch.hub.load('ultralytics/yolov5', 'custom', force_reload = True , path = f'weights/{weights}.pt').autoshape()

    #if weights:
    #    subprocess.run(['python',f'{weights}/detect.py','--weights',f'./weights/{weights}.pt',
    #    '--source','test_image_split/',
    #    '--project','test_data/',
    #    '--name','test_results_raw/',
    #    '--save-txt'])


    imnames = glob.glob(SAVE_PATH + '/*.jpg')

    global_imgs = []

    h_len = int(max([imname[-7] for imname in imnames]))+1
    w_len = int(max([imname[-5] for imname in imnames]))+1

    max_height = img.shape[0]
    max_width = img.shape[1]

    unique_images = set([imname[:-8] for imname in imnames]) 
    num_unique_images = len(unique_images) 

    for i in range(num_unique_images):
        image_name = list(unique_images)[i]

        labels = []

        for j in range(h_len):
            for k in range(w_len):
                im = cv2.imread(image_name + f'_{j}_{k}.jpg')
                res = model(im)
                output = res.pandas().xywh[0]
            
                height = im.shape[0]
                width = im.shape[1]

                output['width'] = output['width']/max_width
                output['height'] = output['height']/max_height

                output['xcenter'] = (k * width + output['xcenter'] )/max_width
                output['ycenter'] = (j * height + output['ycenter'] )/max_height

                labels.append(output)

        merged = pd.concat(labels)
    
    #st.write(merged.to_json(orient='records'))

    

    classes = ['fertilized egg','unfertilized egg','fish larva','unidentifiable object']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    im2 = np.ascontiguousarray(img, dtype=np.uint8)
    height ,width, _ = im2.shape



    for index,row in merged.iterrows():

        # Split string to float
        x, y, w, h = row['xcenter'],row['ycenter'],row['width'],row['height']

        # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
        # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
        x_center, y_center, w, h = float(x)*width, float(y)*height, float(w)*width, float(h)*height
        x1 = round(x_center-w/2)
        y1 = round(y_center-h/2)
        x2 = round(x_center+w/2)
        y2 = round(y_center+h/2) 

        class_idx = row['class']
        

        plot_one_box([x1,y1,x2,y2], im2, color=colors[class_idx], label=classes[class_idx], line_thickness=None)

    st.image(im2)
    #st.dataframe(merged)

    ## include a print of the bounding boxes



        



        ## after display, delete dir
        #READ_PATH = 'test_data/test_results_raw/'
        #output = stitch_test_images(READ_PATH)

        #st.image(output)

    counts = generate_test_results(merged)

    st.dataframe(counts)

        #st.download_button('Download Image',output,file_name = 'inference.jpg')
        #st.download_button('Download Image Counts',counts,file_name = 'counts.csv')


    file_paths += ['test_results_raw/','test_image_split/']
    for file_path in file_paths:
        if os.path.exists(file_path):
            shutil.rmtree(file_path)

