#!/usr/bin/env python3

'''
The work is heavily inspired by the following works:
    - https://www.pyimagesearch.com/2017/07/24/bank-check-ocr-with-opencv-and-python-part-i
    - https://www.datacamp.com/community/tutorials/tensorflow-tutorial
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

     # create a clone of the original image so we can draw on it
    clone = np.dstack([ref.copy()] * 3)
    symbols = []

    # loop over the (sorted) contours
    for index, c in enumerate(refCnts):

        # compute the bounding box of the contour and draw it on our image
        (x, y, w, h) = cv2.boundingRect(c)
        #padding = 2
        padding = 1
        _croped = ref[y-padding:y+h+padding, x-padding:x+w+padding]
        _croped = cv2.resize(_croped, (50, 50))
        cv2.imwrite('images/{0}-captcha.png'.format(index), _croped)
        symbols.append(_croped)

        # coloring the boxes
        # cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output of applying the simple contour method
    # cv2.imshow("Simple Method", clone)
    # cv2.imshow('SimpleImageShower-ref', ref)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return symbols

def load_data(data_directory):
    ''' Load data '''
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels, images = [], []
    for d in directories:
        #labels.append(int(d))
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".png")]
        #print(file_names)
        for f in file_names:
            #images.append(skimage.data.imread(f))
            _image = cv2.imread(f)
            _image_gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
            #_image_gray = cv2.normalize(_image_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(_image_gray)
            labels.append(int(d))

    return images, labels

def train_model_tf(images, labels, symbols_on_captcha):

    import tensorflow as tf

    # Initialize placeholders
    x = tf.placeholder(dtype = tf.float32, shape = [None, 50, 50])
    y = tf.placeholder(dtype = tf.int32, shape = [None])

    # Flatten the input data
    images_flat = tf.contrib.layers.flatten(x)

    # Fully connected layer
    logits = tf.contrib.layers.fully_connected(images_flat, 10, tf.nn.relu)


    # Define loss
    diff = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(diff)

    # Define an optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # Convert logits to label indexes
    correct_pred = tf.argmax(logits, 1)

    # Define an accuracy metric
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", correct_pred)

    tf.set_random_seed(11)

    #images = np.array(images)

    #images = images / 255.0

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train
    for i in range(300):
        #_, accuracy_value = sess.run([train_op, accuracy], feed_dict={x: images, y: labels})
        _, accuracy_value = sess.run([train_op, accuracy], feed_dict={x: images, y: labels})
        if i % 10 == 0 and i != 0:
            print("iteration: {}, accuracy: {} ".format(i, accuracy_value))

    # Pick some random images
    sample_indexes = random.sample(range(len(images)), 10)
    #sample_indexes = [x for x in range(0, len(images),6)]
    sample_images = [images[i] for i in sample_indexes]
    sample_labels = [labels[i] for i in sample_indexes]
    sample_labels = np.array(sample_labels)

    # for value in sample_images:
    #     cv2.imshow('SimpleImageShower', value)
    #     cv2.waitKey(0)

    # Run the "predicted_labels
    predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

    # Print the real and predicted labels
    print(sample_labels)
    print(predicted)

    # calculate accuracy of the test sample
    diff = predicted - sample_labels
    accuracy_cal = (len(diff)-np.count_nonzero(diff)) / len(diff)
    print (accuracy_cal)
    #print(images[0])

    for index, symbol in enumerate(symbols_on_captcha):
        symbol = cv2.normalize(symbol, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #symbol = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)
        # _image = cv2.imread('images/{0}-captcha.png'.format(index))
        # _image_gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        # symbol = _image_gray
        predicted = sess.run([correct_pred], feed_dict={x: [symbol]})[0]
        print(predicted)
        cv2.imshow('SimpleImageShower', symbol)
        cv2.waitKey(0)

    sess.close()

def train_model_tf_v2(images, labels, symbols_on_captcha):
    import tensorflow as tf
    # load data
    # meta, train_data, test_data = input_data.load_data(FLAGS.data_dir, flatten=True)
    # print('data loaded')
    # print('train images: %s. test images: %s' % (train_data.images.shape[0], test_data.images.shape[0]))

    LABEL_SIZE = 10
    IMAGE_SIZE = 50 * 50
    print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))

    # variable in the graph for input data
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
    y_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])

    # define the model
    W = tf.Variable(tf.zeros([IMAGE_SIZE, LABEL_SIZE]))
    b = tf.Variable(tf.zeros([LABEL_SIZE]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(diff)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # forward prop
    predict = tf.argmax(y, axis=1)
    expect = tf.argmax(y_, axis=1)

    # evaluate accuracy
    correct_prediction = tf.equal(predict, expect)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    images_resized = []
    for image in images:
        #image = tf.image.per_image_standardization(image)
        #image = image / 255.0
        image = image.reshape(50 * 50)
        images_resized.append(image)
    images_resized = np.array(images_resized)
    print(images_resized.shape)

    labels_resized = []
    for index, label in enumerate(labels):
        labels_resized.append([0 for x in range(10)])
        labels_resized[index][label] = 1
        #if label == 5: print(labels_resized[index])
    labels_resized = np.array(labels_resized)
    print(labels_resized.shape)
    print(labels_resized[0])

    MAX_STEPS = 10
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Train
        for i in range(MAX_STEPS):
            #batch_xs, batch_ys = train_data.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: images_resized, y_: labels_resized})

            if i % 10 == 0:
                # Test trained model
                r = sess.run(accuracy, feed_dict={x: images_resized, y_: labels_resized})
                print('step = %s, accuracy = %.2f%%' % (i, r * 100))

        # final check after looping
        # r_test = sess.run(accuracy, feed_dict={x: test_data.images, y_: test_data.labels})
        # print('testing accuracy = %.2f%%' % (r_test * 100, ))

        input_resized = []
        for image in symbols_on_captcha:
            #image = image / 255.0
            image = image.reshape(50 * 50)
            input_resized.append(image)
        input_resized = np.array(input_resized)
        print(input_resized.shape)

        for image in input_resized:
            predicted = sess.run(tf.argmax(y, axis=1), feed_dict={x: [image]})
            print(predicted)

def train_model_keras_mnist(images, labels, symbols_on_captcha, retrain = True):

    import tensorflow as tf
    #from keras.layers import Input, Flatten

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

        #model.fit(images, labels, epochs=1)
        model.fit(x_train, y_train, epochs=1)
        tf.keras.models.save_model(model, MODEL_NAME)
    else:
        model = tf.keras.models.load_model(MODEL_NAME)

    # images_resized = []
    # for image in images:
    #     _image = cv2.resize(image, (28, 28))
    #     images_resized.append(_image)

    # # v1 - doesn't work properly
    # for index, image in enumerate(images_resized):
    #     _x = np.expand_dims(image, axis=0)
    #     _x = np.vstack([_x])
    #     predicted = model.predict(_x)
    #     print(predicted)
    #     print(labels[index])

    # # v2 - doesn't work properly
    # images_resized  = np.array(images_resized)
    # for index, image in enumerate(images_resized):
    #     predicted = model.predict(image)
    #     print(predicted)
    #     print(labels[index])
    # #     print(type(image))

    #     cv2.imshow('SimpleImageShower', image)
    #     cv2.waitKey(0)
    #     print ('predicted: {}, real: {}'.format(predicted, labels[index]))

    for index, symbol in enumerate(symbols_on_captcha):
        #symbol = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)
        _image = cv2.imread('images/{0}-captcha.png'.format(index))
        _image = cv2.normalize(_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #_image_gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        _image_gray = cv2.resize(_image_gray, (28, 28))
        symbol = _image_gray
        #symbol = np.array(symbol)
        #symbol = symbol.flatten()
        # symbol = tf.keras.layers.Input(symbol)
        # symbol = tf.keras.layers.Flatten()(symbol)
        #print(symbol.shape)
        symbol = np.expand_dims(symbol, axis=0)
        symbol = np.vstack([symbol])
        #symbol = symbol / 255.0
        predicted = model.predict_classes(symbol, batch_size=1, verbose=1)
        print(predicted)
        # cv2.imshow('SimpleImageShower', _image)
        # cv2.waitKey(0)

def main(args):

    symbols = fetch_symbols_from_captcha(args['image'])

    train_dir = 'c:\\repositories\\pycaptcha\\train\\'
    images, labels = load_data(train_dir)

    #train_model_tf(images, labels, symbols)
    #train_model_tf_v2(images, labels, symbols)
    train_model_keras_mnist(images, labels, symbols)


if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(ap.parse_args())

    main(args)
