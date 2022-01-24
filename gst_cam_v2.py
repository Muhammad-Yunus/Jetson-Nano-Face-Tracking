def camera(i, w, h, fs=21, w0=1280, h0=720):
    return "nvarguscamerasrc sensor_id=%d ! \
    video/x-raw(memory:NVMM), \
    width=%d, height=%d, \
    format=(string)NV12, \
    framerate=%d/1 ! \
    nvvidconv \
    flip-method=2  ! \
    video/x-raw, \
    width=%d, height=%d, \
    format=(string)BGRx ! \
    videoconvert ! \
    video/x-raw, \
    format=(string)BGR ! \
    appsink sync=0" % (i, w0, h0, fs, w, h)