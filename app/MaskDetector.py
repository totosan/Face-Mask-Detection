# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import tensorflow as tf 
import numpy as np
import cv2
import imutils
import os

_Graph=None
_Net=None
_Model=None
_Session=None

class DetectMask():
    def __init__(self,
                 pathFaceDetector='face_detector',
                 model='mask_detector.model',
                 confidence=0.5):
        self.PathToFaceDetectorFolder = pathFaceDetector
        self.ModelFile = model
        self.Confidence = confidence
  
        # load our serialized face detector model from disk
        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join(
            [self.PathToFaceDetectorFolder, "deploy.prototxt"])
        weightsPath = os.path.sep.join(
            [self.PathToFaceDetectorFolder, "res10_300x300_ssd_iter_140000.caffemodel"])
        global _Net
        _Net = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the face mask detector model from disk
        print("[INFO] loading face mask detector model...")
        tfConfig = tf.ConfigProto()
        global _Session
        _Session = tf.Session(config=tfConfig)
        global _Graph
        _Graph = tf.get_default_graph()
        set_session(_Session)
        global _Model    
        _Model = load_model(self.ModelFile)


    def Detect(self, frame):  
        #resize for performance
        frame = imutils.resize(frame, width=300)
        
        # load the input image from disk, clone it, and grab the image spatial
        # dimensions
        (h, w) = frame.shape[:2]
                
        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        print("[INFO] computing face detections...")

        global _Net
        _Net.setInput(blob)
        detections = _Net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]
            print("Found detection with confidence " + str(confidence))
            
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > self.Confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                #cv2.rectangle(frame, (startX, startY), (endX, endY), (128,10,10), 2)
                
                if True:
                    # pass the face through the model to determine if the face
                    # has a mask or not
                    global _Graph, _Model, _Session
                    with _Graph.as_default():
                        set_session(_Session)
                        (mask, withoutMask) = _Model.predict(face)[0]

                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    # include the probability in the label
                    label = "{}: {:.2f}%".format(
                        label, max(mask, withoutMask) * 100)

                    # display the label and bounding box rectangle on the output
                    # frame
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                return frame
