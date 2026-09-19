import cv2
cam = cv2.VideoCapture(1)

ret, frame = cam.read()
count = 0

cv2.imwrite("frame%d.jpg" % count, frame)
count +=1