Fish Larva Detection
==============

Authors: [ChengLing]Cheng Ling, Bhavesh, Leona, Siew Ning
--------------

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/chuachengling/fish-larva-detection/blob/main/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):

```bash
$ git clone https://github.com/chuachengling/fish-larva-detection
$ cd fish-larva-detection
$ pip install -r requirements.txt
```

<details open>
<summary>Installing YoloV5</summary>

```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```


<details open>
<summary>Running the Model</summary>


On 1 terminal run:

```python
python backend/app.py
```

On another terminal run:

```python
streamlit run frontend/frontend_app.py
```

## <div align="center">API Documentation</div>


<p align="left"><img width="800" src="https://github.com/chuachengling/fish-larva-detection/blob/b5d2b54b10b3e685da5b9152c2f23831b980188a/api_image.png"></p>

Endpoints were made on flask and the API has been dockerized. The docker image has been pushed into AWS Elastic Container Registry. A FARGATE instance was run and the port was exposed to the open internet. A frontend application was made using streamlit and is currently hosted on heroku. The frontend is pointed towards the API gateway to access the data.

## <div align="center">Inference using Streamlit</div>


<p align="left"><img width="800" src="https://github.com/chuachengling/fish-larva-detection/blob/a30a1a13b3c875e8335f6bd8df131843cedc8ac3/demo_app.png"></p>

A test image is to be uploaded (png, jpg, jpeg) format and the model will run using the app.py in the terminal. The resulting inferred image, along with the relevant label counts will be generated.

A download button will prompt the user to download the corresponding JSON file.
















