from flask import Flask,request,jsonify
import cv2
import numpy as np
import base64
import io
import os 

from backend.inference import load_model, inference, draw_bboxes

model = load_model()
app = Flask(__name__)

@app.route('/api/',methods =  ['POST'])
def main():
    response = request.get_json()
    data_str = response['image']
    point = data_str.find(',')
    base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"

    image = base64.b64decode(base64_str)       
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    #if(img.mode!='RGB'):
    #    img = img.convert("RGB")

    img_arr = np.array(img)

    results = inference(img_arr,model)
    print(results)
    img_with_bbox = draw_bboxes(img_arr,results)

    ## convert labels and image here 
    return results,img_with_bbox

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')