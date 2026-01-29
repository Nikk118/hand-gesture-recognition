import cv2
import mediapipe as mp
import os

GESTURE = "ok"
SAVE_DIR = f"../images/{GESTURE}"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
count = 0

with mp_hands.Hands() as hands:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Bounding box
            xs = [int(lm.x * w) for lm in hand.landmark]
            ys = [int(lm.y * h) for lm in hand.landmark]

            x_min, x_max = max(min(xs)-20, 0), min(max(xs)+20, w)
            y_min, y_max = max(min(ys)-20, 0), min(max(ys)+20, h)

            hand_crop = frame[y_min:y_max, x_min:x_max]

        cv2.imshow("Press C to Save | Q to Quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and result.multi_hand_landmarks:
            cv2.imwrite(f"{SAVE_DIR}/{count}.jpg", hand_crop)
            print(f"Saved image {count}")
            count += 1

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
