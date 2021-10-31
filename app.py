from flask import Flask,request,jsonify, Response 
import cv2
import numpy as np
import base64
import io
import os 

from backend.inference import *
import backend.config as config


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

@app.route('/api/',methods =  ['POST'])
def main():
    response = request.get_json()
    filename = response['filename']
    data_str = response['image_base64']
    image = base64.b64decode(data_str)

    jpg_as_np = np.frombuffer(image, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    #if(img.mode!='RGB'):
    #    img = img.convert("RGB")

    #img_arr = np.array(img)


    results = inference(img,640)

    img_with_bbox = draw_bboxes(img,config.CLASSES,config.COLORS,results)

    results_list = transform_results(results)

    img_with_bbox = base64.b64encode(img_with_bbox).decode('utf8')
    response = {"filename":filename,
                "image_base64":img_with_bbox,
                "predictions" : results_list}

    ## convert labels and image here 

    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')