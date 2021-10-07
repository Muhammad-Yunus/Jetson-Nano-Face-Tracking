from gst_cam import camera
import cv2 

window_name = "Stream Output"
cv2.namedWindow(window_name, cv2.WINDOW_OPENGL)

w,h = 1280,720
cap = cv2.VideoCapture(camera(0, w, h))

cuImg = cv2.cuda_GpuMat()
cuImg.create((w, h), cv2.CV_8UC3)
cuImgBGR = cv2.cuda_GpuMat()
cuImgBGR.create((w, h), cv2.CV_8UC3)

try :
    while cap.isOpened() :
        e1 = cv2.getTickCount()
        cuImg.upload(cap.read()[1])

        cv2.cuda.cvtColor(cuImg, cv2.COLOR_RGB2BGR, cuImgBGR)

        cv2.imshow(window_name, cuImgBGR)

        if cv2.waitKey(1) == ord("q"):
            break

        e2 = cv2.getTickCount()
        time = (e2 - e1)/ cv2.getTickFrequency()
        print("execution time %.4fs (FPS %.2f)" % (time, (1/time)))

finally :
    cap.Close()
    cv2.destroyAllWindows()