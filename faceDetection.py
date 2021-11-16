import cv2
import mediapipe as mp
import time

class faceDetection():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(min_detection_confidence, model_selection)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin*ih), \
                       int(bboxC.width *iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])

                cv2.rectangle(img, bbox, (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0]*100)}%',
                            (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
        return img, bboxs


def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2 .VideoCapture("pose-video/running.mp4")
    pTime = 0
    detector = faceDetection()
    while True:
        success, img = cap.read()
        scale_percent = 20  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        img, bboxs = detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {(int(fps))}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("image", img)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()