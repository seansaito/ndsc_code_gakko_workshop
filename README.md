# Shopee National Data Science Challenge - Code Gakko Workshop

This repository hosts code presented during Code Gakko's workshop on object detection for the 
[National Data Science Challenge](https://careers.shopee.sg/ndsc/) hosted by Shopee.

## Setting up

1. Go to [this](https://drive.google.com/file/d/1FYrOiKQMORSgldvPUyf68igf1Q9looSb/view?usp=sharing) Google Drive folder and download
the file `yolo.h5` (about 250 MB) into the `model_data` directory.
2. Download the dependencies specified in `requirements.txt` via:

```angular2html
$ pip install -r requirements.txt
```

## Running the YOLO detector

Assuming you have followed the above instructions successfully, you can run the detection program
by simply running the `localize_image.py` module:

```angular2html
$ python localize_image.py
```

#### Usage

```
usage: localize_image.py [-h] [--img_path IMG_PATH]
                         [--detection_score_threshold DETECTION_SCORE_THRESHOLD]
                         [--nms_iou_threshold NMS_IOU_THRESHOLD]

YOLO detection program for images

optional arguments:
  -h, --help            show this help message and exit
  --img_path IMG_PATH   Path to the input image
  --detection_score_threshold DETECTION_SCORE_THRESHOLD
                        The minimum score required for a detection to be a
                        considered
  --nms_iou_threshold NMS_IOU_THRESHOLD
                        The threshold for pruning away bounding box
                        predictions that have overlap with previous selections

```
