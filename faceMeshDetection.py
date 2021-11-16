import cv2
import mediapipe as mp
import time

class faceMeshDetection():
    def __init__(self, static_image_mode=False,
                 max_num_faces=2,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces, self.refine_landmarks, self.min_detection_confidence, self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=0, circle_radius=0)

    def findFaces(self, img):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2 .VideoCapture("pose-video/running.mp4")
    pTime = 0
    detector = faceMeshDetection()
    while True:
        success, img = cap.read()
        scale_percent = 20  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        img, faces = detector.findFaces(img)
        print(len(faces))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {(int(fps))}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("image", img)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()