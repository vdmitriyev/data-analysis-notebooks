#!/usr/bin/env python3

'''
The work is heavily inspired by the following works:
    - https://www.pyimagesearch.com/2017/07/24/bank-check-ocr-with-opencv-and-python-part-i
    - https://www.datacamp.com/community/tutorials/tensorflow-tutorial

Usage:
    $ python symbols_extractor.py --image images\images-with-digits.png
'''

import os
import cv2
import random
import imutils
import argparse
import numpy as np
from imutils import contours

def fetch_symbols_from_captcha(image_path):
    ''' Fetch symbols from a given captcha'''

     # load the image
    image = cv2.imread(image_path)

    # convert it to grayscale, blur it slightly, then find edges
    ref = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find contours in the images
    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

    symbols = [] # collection of images
    padding = 1  # padding for cutting images

    # loop over the (sorted) contours
    for index, c in enumerate(refCnts):
        # compute the bounding box of the contour and draw it on our image
        (x, y, w, h) = cv2.boundingRect(c)

        # crop image
        _croped = ref[y-padding:y+h+padding, x-padding:x+w+padding]
        _croped = cv2.resize(_croped, (30, 30))

        cv2.imwrite('images/{0}-captcha.png'.format(index), _croped)
        symbols.append(_croped)

    # show the output of applying the simple contour method
    # cv2.imshow("Simple Method", ref)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return symbols

def load_data(data_directory):
    ''' Load images from the given repository'''

    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels, images = [], []
    for d in directories:

        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".png")]

        padding = 1
        for f in file_names:
            _image = cv2.imread(f)
            _image_gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
            _image_gray = cv2.threshold(_image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # image must be processed in the save way as the one from captcha
            _ref_cnts = cv2.findContours(_image_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            _ref_cnts = _ref_cnts[0] if imutils.is_cv2() else _ref_cnts[1]
            _ref_cnts = contours.sort_contours(_ref_cnts, method="left-to-right")[0]
            (x, y, w, h) = cv2.boundingRect(_ref_cnts[0])
            _croped = _image_gray[y-padding:y+h+padding, x-padding:x+w+padding]
            _croped = cv2.resize(_croped, (30, 30))

            images.append(_croped)
            labels.append(int(d))

    return images, labels


def main(args):

    symbols = fetch_symbols_from_captcha(args['image'])

    #train_dir = 'c:\\repositories\\pycaptcha\\train\\'
    #images, labels = load_data(train_dir)

if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(ap.parse_args())

    main(args)
