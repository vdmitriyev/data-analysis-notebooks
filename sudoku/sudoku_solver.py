#!/usr/bin/env python3

__description__ = 'A simple sudoku solver based on computer vision and machine learning.'
'''

The current work is heavily inspired by the followings:
    - https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/
    - https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
    - https://github.com/akshaybahadur21/Digit-Recognizer
'''
import os
import cv2
import imutils
import argparse
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours

def sudoku_matrix(image_path, ml_model):
    ''' Get matrix of sudoku '''

    # load the image, convert it to grayscale, blur it slightly, then find edges
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # find contours in the edge map, then initialize the contour that corresponds to the document
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    docCnt = None

    # ensure that at least one contour was found
    if len(cnts) > 0:

        # sort the contours according to their size in descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # loop over the sorted contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then we can assume we have found the paper
            if len(approx) == 4:
                docCnt = approx
                break

    # apply a four point perspective transform to both the
    # original image and grayscale image to obtain a top-down
    # birds eye view of the paper
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    paper = cv2.resize(paper, (560, 560), 0, 0, interpolation = cv2.INTER_CUBIC)
    warped = cv2.resize(warped, (560, 560), 0, 0, interpolation = cv2.INTER_CUBIC)

    #cv2.imshow("SimpleImageShower-paper", paper)
    #cv2.imshow("SimpleImageShower-warped", warped)

    # apply Otsu's thresholding method to binarize the warped piece of paper
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find contours in the thresholded image, then initialize the list of contours that correspond to questions
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    counter = 0
    cnts_srt = contours.sort_contours(cnts[1], method="top-to-bottom")[0]

    for cnt in cnts_srt:

        area = cv2.contourArea(cnt)
        area_coeff = (thresh.shape[0] * thresh.shape[1]) / area
        print (area, area_coeff)

        #if 3400.0 <= area and area <= 3900.0: # pure area based
        if 85.0 <= area_coeff and area_coeff <= 91.0:

            counter += 1
            color = [2*counter, counter, 255]
            cv2.drawContours(paper, [cnt], -1, color, 3)

            [x, y, w, h] = cv2.boundingRect(cnt)
            _cropped = thresh[y:y+h,x:x+w]
            cv2.imwrite('data/{}.png'.format(str(counter).zfill(2)), _cropped)

            _cropped = cv2.resize(_cropped, (28, 28))
            #_reshaped = _cropped.reshape(28 * 28)
            _reshaped = _cropped
            _digit = str(recognize_digit(_reshaped, ml_model))

            # marking image with predicted digits
            cv2.putText(paper, _digit, (x,y+h), 0, 1, (0,255,0))

    cv2.imshow("SimpleImageShower-newImage", paper)
    cv2.waitKey(0)

def recognize_digit(image_flatt, ml_model):
    from numpy import newaxis
    #print(ml_model.input_shape)
    #print(image_flatt.shape)
    image_flatt = image_flatt[newaxis,:,:]
    predicted = ml_model.predict_classes(image_flatt, batch_size=1, verbose=1)
    return predicted[0]

def train_model_keras_mnist(retrain=True ):

    import tensorflow as tf

    mnist = tf.keras.datasets.mnist
    MODEL_NAME = 'keras-mnist-model.h5'

    if not os.path.exists(MODEL_NAME) or retrain:
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(512, activation=tf.nn.relu),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=1)
        tf.keras.models.save_model(model, MODEL_NAME)
    else:
        model = tf.keras.models.load_model(MODEL_NAME)

    return model

def main(args):

    ml_model = train_model_keras_mnist()
    sudoku_matrix(args["image"], ml_model)

if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(ap.parse_args())

    main(args)
