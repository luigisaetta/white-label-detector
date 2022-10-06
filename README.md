# "White Label" Detector
This repository contains most of the work done to develop a "white label" detector on images. It is based on YOLO v5.

## Setup
To use the code for the detector you need to download the file with the trained PyTorch model, in a local directory.

You can get the file from [here](https://objectstorage.eu-frankfurt-1.oraclecloud.com/n/frqap2zhtzbe/b/barcode_models/o/best_label_bianca_yolov5x_80ep.pt).

Then, when you instantiate the Detector class, you have to specify the pathname for the pt file.

See, as an example, this [NB](https://github.com/luigisaetta/white-label-detector/blob/main/test_white_label_detector.ipynb).

## Usage
If you want to get BB rectangles:
```
from white_label_detector import WhiteLabelDetector

# instantiate
detector = WhiteLabelDetector(MODEL_PATH, CONFIDENCE)

# read the image
img1 = read_image("./img1.jpg")

# get a vector with all BB, confidence and classes
boxes = detector.detect_white_labels(img1)

```
If you want to get cropped images:
```
from white_label_detector import WhiteLabelDetector

# instantiate
detector = WhiteLabelDetector(MODEL_PATH, CONFIDENCE)

# read the image
img1 = read_image("./img1.jpg")

# get a list with croped imgs as np array H,W,C
imgs = detector.detect_and_crop_white_labels(img1)

```

## Dependencies
Obviously, you don't need to download the code for YOLO V5 to use the Detector.

The only dependencies are:
* **Torch**, version 1.10.0 or above
* **CV2**, version 4.6.0 or above

The versions listed here are the versions I have used for dev/test.
