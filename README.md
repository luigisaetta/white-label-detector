# "White Label" Detector
This repository contains most of the work done to develop a "white label" detector. It is based on YOLO v5

## Setup
To use the code for the barcode/qrcode detector you need to download the file with the trained PyTorch model, in a local directory.

You can get the file from [here](https://objectstorage.eu-frankfurt-1.oraclecloud.com/n/frqap2zhtzbe/b/barcode_models/o/best_label_bianca_yolov5x_80ep.pt).

Then, when you instantiate the Detector class, you have to specify the pathname for the pt file.

See, as an example, this NB.