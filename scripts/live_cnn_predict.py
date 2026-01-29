import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


MODEL_PATH = "models/gesture_cnn.h5"
GESTURES = ["fist", "ok", "palm", "peace","rock"]
IMAGE_SIZE = 64


model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded")


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands() as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Bounding box
            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in hand.landmark]
            ys = [int(lm.y * h) for lm in hand.landmark]

            x_min, x_max = max(min(xs)-20, 0), min(max(xs)+20, w)
            y_min, y_max = max(min(ys)-20, 0), min(max(ys)+20, h)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                hand_img = cv2.resize(hand_img, (IMAGE_SIZE, IMAGE_SIZE))
                hand_img = hand_img / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                pred = model.predict(hand_img, verbose=0)
                class_id = np.argmax(pred)
                gesture = GESTURES[class_id]

                cv2.putText(
                    frame,
                    gesture,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("CNN Gesture Prediction", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
