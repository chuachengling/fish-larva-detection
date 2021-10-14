import streamlit as st 
from utils import *
import glob
import math
import cv2
import pandas as pd
import subprocess
import os
import shutil
from PIL import Image
import numpy as np


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
    img = Image.open(file)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


    cut_test_images(img,SAVE_PATH,SLICE_SIZE)

    ## RUN INFERENCE 

    weights = st.selectbox('Please choose the weights you want', ['yolov5'])
    if weights:
        subprocess.run(['python',f'{weights}/detect.py','--weights',f'./weights/{weights}.pt',
        '--source','test_image_split/',
        '--project','test_data/',
        '--name','test_results_raw/',
        '--save-txt'])


        ## after display, delete dir
        READ_PATH = 'test_data/test_results_raw/'
        output = stitch_test_images(READ_PATH)

        st.image(output)

        counts = generate_test_results(READ_PATH)

        st.dataframe(counts)

        #st.download_button('Download Image',output,file_name = 'inference.jpg')
        #st.download_button('Download Image Counts',counts,file_name = 'counts.csv')





        file_paths += ['test_results_raw/']
        for file_path in file_paths:
            if os.path.exists(file_path):
                shutil.rmtree(file_path)
