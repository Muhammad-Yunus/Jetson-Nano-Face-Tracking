import cv2
from gst_cam import camera
from flask import Flask, render_template, Response

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

app = Flask(__name__)

# initialize camera (via gstreamer pipeline)
w, h = 1280,720 #480, 320
cap = cv2.VideoCapture(camera(1, w, h))

def detect_face(frame):
    e1 = cv2.getTickCount()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5, minSize=(20,20) )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 2)
    
    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    cv2.putText(frame, 
                "%d FPS - Cascade CPU" % (1/time), 
                (20, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

def gen_frames():  
    while True:
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