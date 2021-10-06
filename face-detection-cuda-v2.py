import cv2
from gst_cam import camera

# initialize camera (via gstreamer pipeline)
w, h = 480, 320
cap = cv2.VideoCapture(camera(1, w, h))
cuFrame = cv2.cuda_GpuMat()
cuFrame.create((w, h), cv2.CV_8UC3)
cuGray = cv2.cuda_GpuMat()
cuGray.create((w, h), cv2.CV_8UC1)
obj_buf = cv2.cuda_GpuMat()

# Create faceDetector object from CUDA CascadeClassifier
xml_face = 'haarcascades/haarcascade_frontalface_default_cuda.xml'
faceDetector = cv2.cuda.CascadeClassifier_create(xml_face)
faceDetector.setScaleFactor(1.5)
faceDetector.setMinNeighbors(5)
faceDetector.setMinObjectSize((20, 20))

while cap.isOpened():
    e1 = cv2.getTickCount()
    ret, frame = cap.read()
    if not ret: 
        break
    
    cuFrame.upload(frame)

    cv2.cuda.cvtColor(cuFrame, cv2.COLOR_BGR2GRAY, cuGray)

    obj_buf = faceDetector.detectMultiScale(cuGray)
    result = obj_buf.download()

    # Draw a rectangle around the faces
    if result is not None:
        for (x, y, w, h) in result[0]:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

     # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(5) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    print("execution time %.4fs" % time)

cv2.destroyAllWindows()
cap.release()