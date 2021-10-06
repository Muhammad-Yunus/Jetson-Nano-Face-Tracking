import jetson.utils
import cv2 

window_name = "Stream Output"
cv2.namedWindow(window_name, cv2.WINDOW_OPENGL)

cap = jetson.utils.videoSource("/dev/video2") 
w,h = cap.GetWidth(), cap.GetHeight()

cuImg = cv2.cuda_GpuMat()
cuImg.create((w, h), cv2.CV_8UC3)
cuGray = cv2.cuda_GpuMat()
cuGray.create((w, h), cv2.CV_8UC1)

try :
    while cap.IsStreaming :
        e1 = cv2.getTickCount()
        cuImg.upload(jetson.utils.cudaToNumpy(cap.Capture()))

        cv2.cuda.cvtColor(cuImg, cv2.COLOR_BGR2GRAY, cuGray)

        cv2.imshow(window_name, cuImg)

        if cv2.waitKey(1) == ord("q"):
            break

        e2 = cv2.getTickCount()
        time = (e2 - e1)/ cv2.getTickFrequency()
        print("execution time %.4fs" % time)

finally :
    cap.Close()
    cv2.destroyAllWindows()