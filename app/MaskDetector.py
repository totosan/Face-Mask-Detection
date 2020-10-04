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
import signal

_FINISH=False

class DetectMask():
    def __init__(self,
                 pathFaceDetector='face_detector',
                 #model='mask_detector.model',
                 model='mask_detect.sv',
                 confidence=0.5):
        if "app" in os.getcwd():
            self.cwd="."
        else:
            self.cwd=os.path.join(os.getcwd(),"app")
        self.PathToFaceDetectorFolder = pathFaceDetector
        self.ModelFile = os.path.join(self.cwd, model)
        self.Confidence = confidence
        
        # load our serialized face detector model from disk
        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join(
            [self.cwd,self.PathToFaceDetectorFolder, "deploy.prototxt"])
        weightsPath = os.path.sep.join(
            [self.cwd,self.PathToFaceDetectorFolder, "res10_300x300_ssd_iter_140000.caffemodel"])
        
        try:
            self._Net = cv2.dnn.readNet(prototxtPath, weightsPath)
        except Exception as ex:
            print(ex)


    def worker(self, input_q, output_q):
        def signal_handler(*args):
            print("Hit Ctrl C .. Terminating worker!")    
            global _FINISH
            _FINISH = True
        # load the face mask detector model from disk
        print("[INFO] loading face mask detector model...")
        tfConfig = tf.ConfigProto()
        self._Model = load_model(self.ModelFile)              
        self._Model._make_predict_function()
        self._Session = tf.Session(config=tfConfig)
        #self._Session.run()
        self._Session.run(tf.global_variables_initializer())
        self._Graph = tf.get_default_graph()
        #self._Graph.finalize()
        #set_session(self._Session)
        signal.signal(signal.SIGINT, signal_handler)

        print('Worker started')
        global _FINISH
        while not _FINISH:
            frame = input_q.get()
            # Check frame object is a 2-D array (video) or 1-D (webcam)
            if len(frame) == 2:
                frame_rgb = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
                output_q.put((frame[0], self.Detect(frame_rgb, self._Net, self._Session, self._Graph, self._Model)))
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output_q.put(self.Detect(frame_rgb, self._Net, self._Session, self._Graph, self._Model))
        print('Exited worker')

    def Detect(self, frame, faceNet, tfSess, tfGraph, tfModel):  
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

        faceNet.setInput(blob)
        detections = faceNet.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > self.Confidence:
                print("Found detection with confidence " + str(confidence))
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
                    with tfGraph.as_default():
                        #set_session(tfSess)
                        (mask, withoutMask) = tfModel.predict(face)[0]
                    
                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    # include the probability in the label
                    label = "{}: {:.2f}%".format(
                        label, max(mask, withoutMask) * 100)

                    print(label)
                    # display the label and bounding box rectangle on the output
                    # frame
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        return frame
