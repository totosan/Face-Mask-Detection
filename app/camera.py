from flask import Flask, render_template, Response
import imutils
import cv2
from MaskDetector import DetectMask

app = Flask(__name__)
detector = DetectMask()
detector.LoadNet()

camera = cv2.VideoCapture(1)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
#camera = cv2.VideoCapture("rtsp://admin:845357@192.168.1.14/live/profile.0")
def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:

            frame = imutils.resize(frame, width=400)
            resultframe = detector.Detect(frame)
            try:
                ret, buffer = cv2.imencode('.jpg', resultframe)
                resultframe = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + resultframe + b'\r\n')  # concat frame one by one and show result
            except Exception as e:
                print("Exception - " + str(e))

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
