import cv2
from hand_tracker import HandTracker
from hand_processor import HandProcessor
from gesture_recognizer import GestureRecognizer
from gesture_db import GestureDB

tracker = HandTracker()
processor = HandProcessor()
db = GestureDB()
recognizer = GestureRecognizer(db)

recording = False
sequence_buffer = []
SEQUENCE_LENGTH = 30

print("Controls: [s] save static | [r] start/stop recording dynamic | [q] quit")

while True:
    success, img, results = tracker.get_frame()
    if not success:
        break

    img = tracker.draw(img, results)
    key = cv2.waitKey(1) & 0xFF

    if results.hand_landmarks:
        landmarks = results.hand_landmarks[0]
        normalized = processor.normalize(landmarks, img.shape)
        vector = processor.to_flat_vector(normalized)

        # Save static gesture
        if key == ord('s'):
            name = input("Name this static gesture: ")
            db.save_static(name, vector)
            print(f"Saved static: '{name}'")

        # Record dynamic gesture
        if key == ord('r'):
            if not recording:
                recording = True
                sequence_buffer = []
                print("Recording... press 'r' again to stop")
            else:
                recording = False
                name = input("Name this dynamic gesture: ")
                db.save_dynamic(name, sequence_buffer)
                print(f"Saved dynamic: '{name}' ({len(sequence_buffer)} frames)")

        if recording:
            sequence_buffer.append(vector)
            cv2.putText(img, f"REC {len(sequence_buffer)}", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # --- Live recognition ---
        match = recognizer.match_static(vector)
        if match:
            cv2.putText(img, f"Gesture: {match}", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    fps = tracker.get_fps()
    cv2.putText(img, f"FPS: {fps}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Gesture Vocabulary", img)

    if key == ord('q'):
        break

tracker.release()
cv2.destroyAllWindows()