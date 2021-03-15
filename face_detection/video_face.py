# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 03:40:59 2020

@author: hp
"""
import cv2  # pip install opencv-python
import dlib
import numpy as np

# Use a file on your computer:
# videoCapture = cv2.VideoCapture('video/occlusion.mp4')

# Or use a web cam:
videoCapture = cv2.VideoCapture(0)

# Initialise three separate models
# dlib
detector2 = dlib.get_frontal_face_detector()

# caffe (DNN)
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Haar cascade
classifier2 = cv2.CascadeClassifier('models/haarcascade_frontalface2.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

# Each iteration of the while loop captures a single frame from the capture device (file or webcam)
while(True):
    # Get the next frame
    ret, img = videoCapture.read()

    # If a frame was successfully captured
    if ret == True:
        # Resize image
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        height, width = img.shape[:2]
        img2 = img.copy()
        img3 = img.copy()
        img4 = img.copy()

        # Convert to greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces using dlib and draw bounding boxes
        faces2 = detector2(gray, 1)

        for result in faces2:
            x = result.left()
            y = result.top()
            x1 = result.right()
            y1 = result.bottom()
            cv2.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img2, 'dlib', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Detect faces using caffe (DNN) and draw bounding boxes
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                        1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces3 = net.forward()

        for i in range(faces3.shape[2]):
            confidence = faces3[0, 0, i, 2]
            if confidence > 0.5:
                box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype("int")
                # cv2.rectangle(img3, (x, y), (x1, y1), (0, 0, 255), 2)
                cv2.rectangle(img3, (x, y), (x1, y1), (0, 0, 255), -1)  # -1 fills the rectangle

        cv2.putText(img3, 'dnn', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Detect faces using Haar cascades and draw bounding boxes
        faces4 = classifier2.detectMultiScale(img)

        for result in faces4:
            x, y, w, h = result
            x1, y1 = x + w, y + h
            cv2.rectangle(img4, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img4, 'haar', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Show on the screen
        cv2.imshow("dlib", img2)
        cv2.imshow("dnn", img3)
        cv2.imshow("haar", img4)

        # Exit the loop with the escape key (with one of the video windows active)
        if cv2.waitKey(1) & 0xFF == 27: # esc
            break
    else:
        break

# Release resources
videoCapture.release()
cv2.destroyAllWindows()
