import cv2
from gst_cam import camera
from adafruit_servokit import ServoKit

# initialize camera (via gstreamer pipeline)
w, h = 480, 320
cap = cv2.VideoCapture(camera(0, w, h))
cuFrame = cv2.cuda_GpuMat()
obj_buf = cv2.cuda_GpuMat()

# Create faceDetector object from CUDA CascadeClassifier
xml_face = 'haarcascades/haarcascade_frontalface_default_cuda.xml'
faceDetector = cv2.cuda.CascadeClassifier_create(xml_face)
faceDetector.setScaleFactor(1.05)
faceDetector.setMinNeighbors(5)
faceDetector.setMinObjectSize((20, 20))

#define servo object
srv = ServoKit(channels=16)

# servo wrapper
def pan(pan_degre) :
    srv.servo[0].angle = pan_degre

def tilt(tilt_degre) :
    srv.servo[1].angle = tilt_degre
    
# Frame Size. Smaller is faster, but less accurate.
# Wide and short is better, since moving your head
FRAME_W = 160
FRAME_H = 120

# Default Pan/Tilt for the camera in degrees.
# Camera range is from -90 to 90
cam_pan = 90
cam_tilt = 90

# Turn the camera to the default position
pan(cam_pan)
tilt(cam_tilt)



# capture frames from the camera
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # Detect faces in the image
    cuFrame.upload(frame)
    obj_buf = faceDetector.detectMultiScale(cuFrame)
    result = obj_buf.download()

    # Draw a rectangle around the faces
    if result is not None:
        for (x, y, w, h) in result[0]:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


            # Correct relative to center of image
            turn_x  = float(-(x + w/2 - (FRAME_W/2)))
            turn_y  = float(y + h/2 - (FRAME_H/2))

            # Convert to percentage offset
            turn_x  /= float(FRAME_W/2)
            turn_y  /= float(FRAME_H/2)

            # Scale offset to degrees
            turn_x   *= 10 # VFOV
            turn_y   *= 10 # HFOV
            #print (turn_x)
            #print (turn_y)
            cam_pan  += turn_x
            cam_tilt += turn_y


            # Clamp Pan/Tilt to 0 to 180 degrees
            cam_pan = max(0,min(180,cam_pan))
            cam_tilt = max(0,min(180,cam_tilt))

            # Update the servos
            pan(int(cam_pan))
            tilt(int(cam_tilt))
            cv2.putText(frame, "Pan : " + str(int(cam_pan)) + " tilt: " + str(int(cam_tilt)), (20,20), 	cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()