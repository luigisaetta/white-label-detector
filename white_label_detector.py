""" WhiteLabelDetector

The class encapsulate a YOLO v5 model to detect white and yellow labels 
in images

MIT license
"""

__author__ = "L. Saetta"
__version__ = "0.8"

import torch
import cv2
import numpy as np


class WhiteLabelDetector:
    PARENT_MODEL = "ultralytics/yolov5"
    # if you want to show the images with white labels detectd (for example, in a NB)
    SHOW = False
    # if we want to apply yolov5 TTA in inference (default = True), can be changed
    # without TTA is a little bit faster
    AUGMENT = True

    def _load_model(self):
        self.model = torch.hub.load(self.PARENT_MODEL, "custom", path=self.model_path)
        # and set the confidence level
        self.model.conf = self.confidence

    def __init__(self, model_path, confidence=0.25):
        self.model_path = model_path
        self.confidence = confidence

        self._load_model()

    def _filter_white_labels(self, vet):
        list_out = []

        for row in vet:
            if "bianca" in row[6]:
                list_out.append(row)

        vet_out = np.array(list_out)

        return vet_out

    # reduce the # of decimal digits
    def _reduce_digits(self, vet):
        # for every row
        for row in vet:
            # take the first four columns (xmin, ymin...)
            for i in range(4):
                # remove dec digits
                row[i] = round(row[i], 1)

            # and the fifth (confidence level)
            row[4] = round(row[4], 3)

        return vet

    """this is a utility method.... not to be called from outside """

    def do_crop_for_class(self, results, class_name):
        # class could be barcode, qrcode barcode o qrcode
        # results is what returned from model()

        imgs = results.crop(save=False)

        list_imgs_barcode = []

        for img in imgs:
            if class_name.lower() in img["label"]:
                im = img["im"]
                # trasforma l'immagine in RGB altrimenti icolori sono cambiati
                img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                list_imgs_barcode.append(img_rgb)

        return list_imgs_barcode

    """ the function take as input the img read as np array and returns a matrix
    with one row for every white label detected and
    (xmin, ymin, xmax, ymax, confidence, class#, class)
    This func. returns BB
    """

    def detect_white_labels(self, img: np.ndarray):
        # using TTA
        results = self.model(img, augment=self.AUGMENT)

        if self.SHOW:
            results.print()
            print(results.pandas().xyxy[0])
            results.show()

        # this is the way we get output from yolo v5
        # .values cause I want to work on np array
        vet_boxes = results.pandas().xyxy[0].values

        # take only the white labels
        if vet_boxes.shape[0] > 0:
            vet_boxes = self._filter_white_labels(vet_boxes)

        # some processing to remove some decimal digits
        if vet_boxes.shape[0] > 0:
            vet_boxes = self._reduce_digits(vet_boxes)

        # this way we return a array with one row for detected label
        # (xmin, ymin, xmax, ymax, confidence, class#, class)

        # consider that labels are returned in order of decreasing confidence, not by location
        # but you have the BB coords... so you can establish which is above and which is below
        return vet_boxes

    """ This func returns a list of imgs (as np array, H,W,C, RGB) """

    def detect_and_crop_white_labels(self, img: np.ndarray):
        # using TTA
        results = self.model(img, augment=self.AUGMENT)

        # first the barcodes and then qrcodes
        list_white_labels = self.do_crop_for_class(results, "bianca")

        return list_white_labels
