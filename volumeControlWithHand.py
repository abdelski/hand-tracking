import cv2
import mediapipe
import time
import numpy as np
import handTracking as htm
import math
from subprocess import call

#####################################
wCam, hCam = 1080, 720
pTime = 0
volBar = 400
volPercentage = 0
#####################################

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

detector = htm.handTracking()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if (len(lmList) != 0):
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # Hand range 50-300
        # Volume range 0-100
        vol = np.interp(length, [50, 250], [0, 100])
        volBar = np.interp(length, [50, 250], [400, 150])
        volPercentage = np.interp(length, [50, 250], [0, 100])
        print(int(length), vol)
        call(["amixer", "-D", "pulse", "sset", "Master", str(vol) + "%"])

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'VOL: {int(volPercentage)}%', (40, 130), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {(int(fps))}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey(1)
