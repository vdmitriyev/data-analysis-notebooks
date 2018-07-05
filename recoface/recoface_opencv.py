#!/usr/bin/env python3

__version__ = '1.0'
__updated__ = '05.07.2018'
__description__ = 'A face recongnition script. The files have to be located into images/<NAME-TO-SHOW>/ directory.'

import os
import cv2
import time
import random
import requests
import datetime
import face_recognition

from pygame import mixer

def get_images():

    images = []

    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    images_dir = os.path.join(ROOT_DIR, 'images')

    for subdir, dirs, files in os.walk(images_dir):
        for file in files:
            images.append(os.path.join(subdir, file))
    return images

def get_known_faces(images):

    known_faces_encodings = []
    known_faces_names = []

    for file in images:
        print ('Loading following image: {0}'.format(file))
        loaded_face = face_recognition.face_encodings(face_recognition.load_image_file(file))
        if len(loaded_face) > 0:
            known_faces_encodings.append(loaded_face[0])
            arr_file = file.split(os.sep)
            known_faces_names.append(arr_file[len(arr_file)-2])
        else:
            print('Ignoring image: {0}'.format(file))
    print(known_faces_names)
    return known_faces_encodings, known_faces_names


def face_detection(face_encoding, known_faces_encodings, known_faces_names):

    name = "Unknown"

    matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
    if True in matches:
        first_match_index = matches.index(True)
        name = known_faces_names[first_match_index]

    return name

# def welcome(type):

#     messages = {
#         'hello': [os.path.join(welcome_folder, file) for file in ['hello_01.wav', 'hello_02.wav', 'hello_03.wav', 'hello_04.wav', 'hello_05.wav']],
#         'welcome': [os.path.join(welcome_folder, file) for file in ['welcome_01.wav', 'welcome_02.wav', 'welcome_03.wav']],
#         'unwelcome': [os.path.join(welcome_folder, file) for file in ['unwelcome_01.mp3', 'unwelcome_02.mp3', 'unwelcome_03.wav', 'unwelcome_04.wav', 'unwelcome_05.wav']],
#         'whoareyou': [os.path.join(welcome_folder, file) for file in ['whoareyou_01.wav', 'whoareyou_02.wav', 'whoareyou_03.wav', 'whoareyou_04.wav']]
#     }
#     mixer.init()
#     mixer.music.load(random.choice(messages[type]))
#     mixer.music.play()

def process(known_faces_encodings, known_faces_names):

    video_capture = cv2.VideoCapture(0)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:

        #print(known_faces_encodings)
        #break

        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:

                # see if the face is a match for the known face(s)
                name = face_detection(face_encoding, known_faces_encodings, known_faces_names)
                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('VideoStream', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # time.sleep(0.5)

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    images  = get_images()
    known_faces_encodings, known_faces_names = get_known_faces(images)
    process(known_faces_encodings, known_faces_names)

if __name__ == '__main__':
    main()
