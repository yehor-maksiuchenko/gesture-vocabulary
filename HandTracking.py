import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
    (5, 9), (9, 13), (13, 17)                 # Palm
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=4
)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Wrap frame for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        
        # Process
        results = landmarker.detect(mp_image)
        #print(results.hand_landmarks)

        if results.hand_landmarks:
            for handLms in results.hand_landmarks:
                h, w, _ = img.shape

                # Draw each of the 21 landmark points
                for lm in handLms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

                # Draw connections manually using the connection list
                for connection in HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start = handLms[start_idx]
                    end = handLms[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


        cv2.imshow("Image", img)
        cv2.waitKey(1)