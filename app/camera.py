import cv2
from flask import Flask, render_template, Response
from MaskDetector import DetectMask
from models.AppConfig import *
import multiprocessing
from multiprocessing import Queue, Pool
from queue import PriorityQueue
from models.app_utils import *

app = Flask(__name__)

detector = DetectMask()

appConfig = AppConfig()
appConfig.LoggingEnabled = True
appConfig.QueueSize = 2

# Set the multiprocessing logger to debug if required
if appConfig.LoggingEnabled:
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

# Multiprocessing: Init input and output Queue, output Priority Queue and pool of workers
input_q = Queue(maxsize=appConfig.QueueSize)
output_q = Queue(maxsize=appConfig.QueueSize)
output_pq = PriorityQueue(maxsize=3*appConfig.QueueSize)
pool = Pool(appConfig.NumWorkers, detector.worker, (input_q,output_q))

#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
#camera = cv2.VideoCapture("rtsp://admin:845357@192.168.1.14/live/profile.0")
videoStream = cv2.VideoCapture('MaskVideo.m4v')  # use 0 for web camera

def gen_frames():  # generate frame by frame from camera
    countReadFrame = 0
    nFrame = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT))
    countWriteFrame = 1
    firstReadFrame = True
    firstTreatedFrame = True
    firstUsedFrame = True
    while True:
        try:
            if not input_q.full():
                # Capture frame-by-frame
                success, frame = videoStream.read()  # read the camera frame
                if not success:
                    print('No success')
                    break

                input_q.put((int(videoStream.get(cv2.CAP_PROP_POS_FRAMES)),frame))
                countReadFrame = countReadFrame + 1
                if firstReadFrame:
                    print(" --> Reading first frames from input file. Feeding input queue.\n")
                    firstReadFrame = False
                
                # Check output queue is not empty
                if not output_q.empty():
                    # Recover treated frame in output queue and feed priority queue
                    output_pq.put(output_q.get())
                    if firstTreatedFrame:
                        print(" --> Recovering the first treated frame.\n")
                        firstTreatedFrame = False

                # Check output priority queue is not empty
                if not output_pq.empty():
                    prior, output_frame = output_pq.get()
                    if prior > countWriteFrame:
                        output_pq.put((prior, output_frame))
                    else:
                        countWriteFrame = countWriteFrame + 1
                        output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                        try:
                            ret, buffer = cv2.imencode('.jpg', output_rgb)
                            resultframe = buffer.tobytes()
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + output_rgb + b'\r\n')  # concat frame one by one and show result
                        except Exception as e:
                            print("Exception - " + str(e))

                        if firstUsedFrame:
                            print(" --> Start using recovered frame (displaying and/or writing).\n")
                            firstUsedFrame = False

            print("Read frames: %-3i %% -- Write frame: %-3i %%" % (int(countReadFrame/nFrame * 100), int(countWriteFrame/nFrame * 100)), end ='\r')
            if((not success) & input_q.empty() & output_q.empty() & output_pq.empty()):
                break

#                resultframe = detector.Detect(frame)
        except Exception as ex:
            print('Video Read Exception: ',ex.message)

    pool.terminate()
    videoStream.release()

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
