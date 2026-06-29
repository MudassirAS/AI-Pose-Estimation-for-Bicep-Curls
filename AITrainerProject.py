import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("data/barbell biceps curl_31.mp4")

detector = pm.PoseDetector()
count = 0
dir = 0


while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    # img = cv2.imread("data/pose.png")

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    #print(lmList)

    if len(lmList) != 0:
        # Right Arm
        detector.findAngle(img, 12, 14, 16)
        # Left Arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        #print(angle, per)

        # Check for the dumbbell curls
        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0
        
        print(count)

        cv2.putText(img, str(count), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

