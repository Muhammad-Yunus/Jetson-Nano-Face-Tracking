import cv2
from gst_cam import camera
from flask import Flask, render_template, Response


app = Flask(__name__)

# initialize camera (via gstreamer pipeline)
w, h = 1280,720 #480, 320
cap = cv2.VideoCapture(camera(0, w, h))

def gen_frames():  
    while True:
        e1 = cv2.getTickCount()
        success, frame = cap.read()
        if not success:
            break
        else:
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            e2 = cv2.getTickCount()
            time = (e2 - e1)/ cv2.getTickFrequency()
            print("execution time %.4fs (FPS %.2f)" % (time, (1/time)))
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