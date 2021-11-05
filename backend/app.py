from flask import Flask,request,jsonify,Response 
import cv2
import numpy as np
import base64
import io
import os 
import json

from inference import *
import config as config


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

@app.route('/api/',methods =  ['POST'])
def main():
    response = request.get_json()

    response = json.loads(response)

    filename = response['filename']
    data_str = response['image_base64']
    image = base64.b64decode(data_str)

    jpg_as_np = np.frombuffer(image, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path = '../weights/yolov5.pt',force_reload=True).autoshape() ## put force_reload=True to redownload.

    results = inference(img,model,640)

    img_with_bbox = draw_bboxes(img,config.CLASSES,config.COLORS,results)
    
    #cv2.imwrite('results.jpg',img_with_bbox)

    results_list = transform_results(results)

    _, imagebytes = cv2.imencode('.jpg', img_with_bbox)
    image_out = base64.b64encode(imagebytes)
    
    response = {"filename":filename,
                "image_base64":image_out,
                "predictions" : results_list}


    ## convert labels and image here 

    return jsonify(**response)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)