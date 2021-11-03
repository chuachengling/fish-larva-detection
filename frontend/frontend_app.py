import streamlit as st 
from utils import *
from collections import Counter
import glob
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
import base64
import requests
import json

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366")
    }
   .sidebar .sidebar-content {
        background: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366")
    }
    </style>
    """,
    unsafe_allow_html=True
)


## Choose weights/model -- yolov5
##if want change weights, change the url to the relevant weights

##File uploader which will go to url

##predict function will send post request to the url we run the api from -- follow format from lect notes

##current url: localhost

## Upload the photo
 
##backend will run the model

st.title('Welcome to DSA4266 Group 1 Fish Larva Detection Application')

file = st.file_uploader('Please upload an image',type = ['png','jpg','jpeg'])

def predict(url, file):
    #image = open(file, 'rb')
    image = base64.b64encode(file.read()).decode("utf8")

    ##img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)[:,:,::-1] -- template code to encode to rgb array

    header = {"content-type": "application/json"}
    data = {"filename": file.name,
    "image_base64": image}

    response = requests.post(url, json=json.dumps(data))#, header=header)
    return response ##json format

def rename_class(row):
    if row['class'] == 0:
      return 'fertilized egg'
    if row['class'] == 1:
      return 'unfertilized egg'
    if row['class'] == 2:
      return 'fish larva'
    if row['class'] == 3:
      return 'unidentifiable object'

if file:
    response = predict('http://0.0.0.0:5000/api/',file)
    print('JSON file has been received')
    st.write(json.loads(response.text))


    #Showcase Inferenced Image
    st.write('### Inferenced Image')

    list_new = []
    for i in response:
        list_new.append(i["category_name"])

    count_model = Counter(list_new)
    
    #Create DF
    df = pd.DataFrame.from_dict(count_model, orient='index').reset_index()
    df = df.rename(columns={"index": "class", 0: "model_count"})
    st.dataframe(df)





##take out image and bounding boxes and run the counter --> display counter table
##implement download button to download json file

##take out image -- decode into numpy array display it --> run the counter
##showcase counts of image for respective labels
##implement download button to get response json object, check streamlit for download button, print out image first and get the json format



