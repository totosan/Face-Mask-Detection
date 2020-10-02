import cv2
from flask import Flask, render_template, Response
from MaskDetector import DetectMask
from models.AppConfig import *
import multiprocessing
from multiprocessing import Queue, Pool
from queue import PriorityQueue
from models.app_utils import *
import argparse
import signal

app = Flask(__name__)
appConfig = AppConfig()
_FINISH=False

# Set the multiprocessing logger to debug if required
if appConfig.LoggingEnabled:
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

def cancelEvent():
    return _FINISH

# grabbing ctrl+c (stop signals)
#original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
def signal_handler(*args):
    print("Hit Ctrl C .. Terminating Pool!")    
    _FINISH = True
    pool.terminate()
    pool.join()

detector = DetectMask(cancelation=cancelEvent)


# Multiprocessing: Init input and output Queue, output Priority Queue and pool of workers
input_q = Queue(maxsize=appConfig.QueueSize)
output_q = Queue(maxsize=appConfig.QueueSize)
output_pq = PriorityQueue(maxsize=3*appConfig.QueueSize)
pool = Pool(appConfig.NumWorkers, detector.worker, (input_q, output_q))
signal.signal(signal.SIGINT, signal_handler)

#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
#videoStream = cv2.VideoCapture("rtsp://admin:845357@192.168.1.14/live/profile.0")
#videoStream = cv2.VideoCapture(0)  # use 0 for web camera
videoStream = WebcamVideoStream(src=0).start()
#videoStream = cv2.VideoCapture('/workspaces/Face-Mask-Detection/app/MaskVideo.m4v')  # use 0 for web camera
isVideoStream = not isinstance( videoStream, WebcamVideoStream) 
fps = FPS().start()

def gen_frames():  # generate frame by frame from camera
    countReadFrame = 0
    nFrame = 0
    if isVideoStream:
        nFrame = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT))
    countWriteFrame = 1
    firstReadFrame = True
    firstTreatedFrame = True
    firstUsedFrame = True
        
    def streamToWeb(firstUsedFrame, frame2stream):
        output_rgb = cv2.cvtColor(frame2stream, cv2.COLOR_RGB2BGR)
        try:
            ret, buffer = cv2.imencode('.jpg', output_rgb)
            resultframe = buffer.tobytes()
#                            cv2.putText(resultframe, fps.fps(), (10, 10 - 10),
#                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,128,0), 2)
            return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + resultframe + b'\r\n')  # concat frame one by one and show result
        except Exception as e:
            print("Exception - " + str(e))
        fps.update()
        if firstUsedFrame:
            print(" --> Start using recovered frame (displaying and/or writing).\n")
            firstUsedFrame = False
    
    while True:
        try:
            if not input_q.full():
                # Capture frame-by-frame
                success, frame = videoStream.read()  # read the camera frame
                if not success:
                    print('No success')
                    break
                if isVideoStream:
                    input_q.put((int(videoStream.get(cv2.CAP_PROP_POS_FRAMES)),frame))
                else:
                    input_q.put(frame)
                countReadFrame = countReadFrame + 1
                if firstReadFrame:
                    print(" --> Reading first frames from input file. Feeding input queue.\n")
                    firstReadFrame = False
                
                # Check output queue is not empty
                if not output_q.empty():
                    
                    if isVideoStream:
                        # Recover treated frame in output queue and feed priority queue
                        output_pq.put(output_q.get())
                        if firstTreatedFrame:
                            print(" --> Recovering the first treated frame.\n")
                            firstTreatedFrame = False
                    else:
                        output_frame = output_q.get();
                        yield streamToWeb(firstReadFrame, output_frame)

                # Check output priority queue is not empty by using Video stream
                if isVideoStream and not output_pq.empty():
                    prior, output_frame = output_pq.get()
                    if prior > countWriteFrame:
                        output_pq.put((prior, output_frame))
                    else:
                        countWriteFrame = countWriteFrame + 1
                        yield streamToWeb(firstReadFrame, output_frame)


            if isVideoStream:
                print("Read frames: %-3i %% -- Write frame: %-3i %%" % (int(countReadFrame/nFrame * 100), int(countWriteFrame/nFrame * 100)), end ='\r')
                if((not success) & input_q.empty() & output_q.empty() & output_pq.empty()):
                    break

#                resultframe = detector.Detect(frame)
        except KeyboardInterrupt:
            print('Received CTRL-C  -> Terminating workers!')
            pool.terminate()
        except Exception as ex:
            print('Video Read Exception: ',ex)
    
    fps.stop()
    pool.terminate()
    videoStream.release()
    pool.close()
    pool.join()

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
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-qs","--queue-size",type=int, default=2, help="Pass size of queues for piping frames")
    ap.add_argument("-w","--worker",type=int, default=2, help="Pass the number of worker analyzing frames from queues")
    args=vars(ap.parse_args())
            
    appConfig.LoggingEnabled = True
    appConfig.QueueSize = args['queue_size']
    appConfig.NumWorkers = args['worker']

    app.run(host='0.0.0.0',port=5001)

