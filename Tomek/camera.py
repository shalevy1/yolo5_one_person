import cv2
import numpy as np
from signal_visualization import signal_visualization

camera_adress = "rtsp://admin:12345678-q@192.168.1.70:554"
camera = cv2.VideoCapture(camera_adress)
iterator = 0

print(camera.isOpened())

while(camera.isOpened()):
    iterator += 1
    ret, frame = camera.read()
    # cv2.line(frame,(0,0),(150,1050),(255,0,0),15)
    cv2.imshow('video output', frame)
    if iterator % 20 == 0:
        signal_visualization(True)
    else:
        signal_visualization(False)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    