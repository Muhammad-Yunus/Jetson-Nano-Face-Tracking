
import cv2
from gst_cam import camera
from flask import Flask, render_template, Response

# initialize camera (via gstreamer pipeline)
w, h = 480, 320
cap = cv2.VideoCapture(camera(1, w, h))
cuFrame = cv2.cuda_GpuMat()
obj_buf = cv2.cuda_GpuMat()

# Create faceDetector object from CUDA CascadeClassifier
xml_face = 'haarcascades/haarcascade_frontalface_default_cuda.xml'
faceDetector = cv2.cuda.CascadeClassifier_create(xml_face)
faceDetector.setScaleFactor(1.5)
faceDetector.setMinNeighbors(5)
faceDetector.setMinObjectSize((40, 40))

# initialize Flask
app = Flask(__name__)

def detect_face(frame):
    e1 = cv2.getTickCount()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cuFrame.upload(frame_gray)
    obj_buf = faceDetector.detectMultiScale(cuFrame)
    result = obj_buf.download()

    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    cv2.putText(frame, 
                "%d FPS - Cascade CUDA GPU" % (1/time), 
                (20, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw a rectangle around the faces
    if result is not None:
        for (x, y, w, h) in result[0]:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

def gen_frames():  
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_face(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host="0.0.0.0")
cap.release()