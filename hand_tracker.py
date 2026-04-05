import cv2
import mediapipe as mp
import time

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

class HandTracker:
    def __init__(self, model_path='hand_landmarker.task', num_hands=2):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=num_hands
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)
        self.prev_time = 0

    def get_frame(self):
        """Returns (success, img, results) for each camera frame."""
        success, img = self.cap.read()
        if not success or img is None:
            return False, None, None

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        results = self.landmarker.detect(mp_image)
        return True, img, results

    def draw(self, img, results):
        """Draws landmarks on the image. Returns annotated image."""
        if not results.hand_landmarks:
            return img
        for handLms in results.hand_landmarks:
            h, w, _ = img.shape
            for lm in handLms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)
            for c in HAND_CONNECTIONS:
                s, e = handLms[c[0]], handLms[c[1]]
                x1,y1 = int(s.x*w), int(s.y*h)
                x2,y2 = int(e.x*w), int(e.y*h)
                cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
        return img

    def get_fps(self):
        now = time.time()
        fps = 1 / (now - self.prev_time) if self.prev_time else 0
        self.prev_time = now
        return int(fps)

    def release(self):
        self.cap.release()