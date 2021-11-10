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
## <div align="center">Inference using Streamlit</div>


<p align="left"><img width="800" src="https://github.com/chuachengling/fish-larva-detection/blob/b5d2b54b10b3e685da5b9152c2f23831b980188a/api_image.png"></p>















