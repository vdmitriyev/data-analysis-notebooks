#!/usr/bin/env python3

'''

The work is heavily inspired by the following works:
    - https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/
    - https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
    - https://github.com/akshaybahadur21/Digit-Recognizer

'''

# import the necessary packages
import cv2
import imutils
import argparse
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours

import input_data
import digit_recognizer_lr as digital_recognizer_lr
import digit_recognizer_nn as digital_recognizer_nn


# # get MNIST train data
# mnist_train = input_data.read_data_sets("data/MNIST/", one_hot=False)
# data_train = mnist_train.train.next_batch(5000)
# train_X = data_train[0]
# Y = data_train[1]
# train_Y = (np.arange(np.max(Y) + 1) == Y[:, None]).astype(int)

# # get MNIST test data
# mnist_test = input_data.read_data_sets("data/MNIST/", one_hot=False)
# data_test = mnist_test.train.next_batch(1000)
# test_Y = data_test[1]
# test_X = data_test[0]

# train LR model
#model_lr = digital_recognizer_lr.model(train_X.T, train_Y.T, Y, test_X.T, test_Y, num_iters=1500, alpha=0.05, print_cost=True)
#model_lr = digital_recognizer_lr.model(train_X.T, train_Y.T, Y, test_X.T, test_Y, num_iters=50, alpha=0.05, print_cost=True)
#w_model_lr = model_lr["w"]
#b_model_lr = model_lr["b"]

# train NN model
#model_nn = digital_recognizer_nn.model_nn(train_X.T, train_Y.T, Y, test_X.T, test_Y, n_h=100, num_iters=1500, alpha=0.05, print_cost=True)
#model_nn = digital_recognizer_nn.model_nn(train_X.T, train_Y.T, Y, test_X.T, test_Y, n_h=100, num_iters=100, alpha=0.05, print_cost=True)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
args = vars(ap.parse_args())

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}


# load the image, convert it to grayscale, blur it
# slightly, then find edges
image = cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None
#print (cnts)

# ensure that at least one contour was found
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # loop over the sorted contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        #print (len(approx))

        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = approx
            break

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

#paper = cv2.resize(paper, (560, 560), 0, 0, interpolation = cv2.INTER_CUBIC)
#warped = cv2.resize(warped, (560, 560), 0, 0, interpolation = cv2.INTER_CUBIC)

#cv2.imshow("SimpleImageShower-paper", paper)
cv2.imshow("SimpleImageShower-warped", warped)

# apply Otsu's thresholding method to binarize the warped
# piece of paper
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

print(thresh.shape)

#######   training part    ###############
#samples = np.loadtxt('generalsamples.data',np.float32)
#responses = np.loadtxt('generalresponses.data',np.float32)
#responses = responses.reshape((responses.size,1))
#samples = np.loadtxt('data/merged/traindata.csv', delimiter=',', dtype=np.float32)
#responses = np.loadtxt('data/merged/labels.csv', delimiter=',', dtype=np.float32)

dataset = '01'
#dataset = 'merged'

samples = np.loadtxt('data/own/{0}/traindata.csv'.format(dataset), delimiter=',', dtype=np.float32)
responses = np.loadtxt('data/own/{0}/labels.csv'.format(dataset), delimiter=',', dtype=np.float32)
responses = responses.reshape((responses.size,1))

unique = set(responses.flat)

knn = cv2.ml.KNearest_create()
knn.train(samples,cv2.ml.ROW_SAMPLE,responses)

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(len(cnts[0]))
print(len(cnts[1]))
print(len(cnts[2]))
#cnts = cnts[0] #if imutils.is_cv2() else cnts[0]
# questionCnts = []
#print (cnts)
#print (len(cnts))
counter = 0

#f_train = open('data/traindata.csv','wb')

cnts_srt = contours.sort_contours(cnts[1], method="top-to-bottom")[0]

import math

# loop over the contours
for cnt in cnts_srt:
    #print ('----------------------------')
    area = cv2.contourArea(cnt)
    #print(area)

    #print (thresh.shape[0] / area, thresh.shape[1] / area)
    #print((thresh.shape[0] * thresh.shape[1]) / area)

    area_coeff = (thresh.shape[0] * thresh.shape[1]) / area
    #print (thresh.shape[1] / 3400, thresh.shape[1] / 3900)

    #if 3400.0 <= area and area <= 3900.0: # pure area based
    if 85.0 <= area_coeff and area_coeff <= 91.0:

        #print('YES')
        counter+=1
        color = [2*counter, counter, 255]
        cv2.drawContours(paper, [cnt], -1, color, 3)

        [x,y,w,h] = cv2.boundingRect(cnt)
        roi = thresh[y:y+h,x:x+w]
        #cv2.imwrite('data/{}.png'.format(str(counter).zfill(2)), roi)

        #print (x,y,w,h)
        #print (roi.shape)

        #cv2.waitKey(0)

        roismall = cv2.resize(roi,(60,60))
        roismall = roismall.reshape((1,3600))
        roismall = np.float32(roismall)

        #np.savetxt(f_train, roismall, fmt='%.5f', delimiter=',', newline='\n')

        #print(roismall)
        retval, results, neigh_resp, dists = knn.findNearest(roismall, k = 1)
        digit = str(int((results[0][0])))
        #print('digit: {0}'.format(digit))
        #cv2.putText(paper,digit,(x,y+h),0,1,(0,255,0))
        #cv2.imwrite('data/own/digits/{0}/X2_{1}.png'.format(digit, str(counter).zfill(2)), cv2.resize(roi,(60,60)))
        #print ((x,y+h))

        # # show
        # cv2.putText(roi, digit,(0,60),0,1,(255,255,255))
        # cv2.imshow("SimpleImageShower-roi", roi)
        # cv2.waitKey(0)

        newImage = cv2.resize(roi, (28, 28))

        newImage_np = np.array(newImage)
        newImage_fl = newImage_np.flatten()
        newImage_rs = newImage_fl.reshape(newImage_fl.shape[0], 1)

        #answer_lr = digital_recognizer_lr.predict(w_model_lr, b_model_lr, newImage_rs)[0]
        #print('newImage (LR): {0}'.format(answer_lr))
        #cv2.putText(paper,str(answer_lr),(x+15,y+h),0,1,(0,0,255))

        # answer_nn = digital_recognizer_nn.predict_nn(model_nn, newImage_rs)[0]
        # print('newImage (NN): {0}'.format(answer_nn))
        # cv2.putText(paper,str(answer_nn),(x+15,y+h),0,1,(0,0,255))

        # show
        #cv2.putText(newImage, str(answer_lr), (0,28),0,1,(255,255,255))
        #cv2.imshow("SimpleImageShower-newImage", newImage)
        #cv2.waitKey(0)

        # the last step
        cv2.putText(paper,digit,(x,y+h),0,1,(0,255,0))
        #cv2.putText(paper,digit,(x,y+h),0,1,(0,255,0))

    #else:
        #print('NO')

        #break
        #print(digit)
        # if (float(digit) not in unique):
        #     print()

        #cv2.waitKey(0)

    #break

    # # compute the bounding box of the contour, then use the
    # # bounding box to derive the aspect ratio
    # (x, y, w, h) = cv2.boundingRect(c)
    # ar = w / float(h)

    # # in order to label the contour as a question, region
    # # should be sufficiently wide, sufficiently tall, and
    # # have an aspect ratio approximately equal to 1
    # if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
    #     questionCnts.append(c)

#f_train.close()
#print(counter)
cv2.imshow("SimpleImageShower-paper", paper)

# for value in questionCnts:
#     print(value)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
# questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
# correct = 0

# # each question has 5 possible answers, to loop over the
# # question in batches of 5
# for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
#     # sort the contours for the current question from
#     # left to right, then initialize the index of the
#     # bubbled answer
#     cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
#     bubbled = None

#     # loop over the sorted contours
#     for (j, c) in enumerate(cnts):
#         # construct a mask that reveals only the current
#         # "bubble" for the question
#         mask = np.zeros(thresh.shape, dtype="uint8")
#         cv2.drawContours(mask, [c], -1, 255, -1)

#         # apply the mask to the thresholded image, then
#         # count the number of non-zero pixels in the
#         # bubble area
#         mask = cv2.bitwise_and(thresh, thresh, mask=mask)
#         total = cv2.countNonZero(mask)

#         # if the current total has a larger number of total
#         # non-zero pixels, then we are examining the currently
#         # bubbled-in answer
#         if bubbled is None or total > bubbled[0]:
#             bubbled = (total, j)
#         # initialize the contour color and the index of the

#     # *correct* answer
#     color = (0, 0, 255)
#     k = ANSWER_KEY[q]

#     # check to see if the bubbled answer is correct
#     if k == bubbled[1]:
#         color = (0, 255, 0)
#         correct += 1

#     # draw the outline of the correct answer on the test
#     cv2.drawContours(paper, [cnts[k]], -1, color, 3)

#cv2.imshow("SimpleImageShower-paper", paper)

cv2.waitKey(0)
cv2.destroyAllWindows()
