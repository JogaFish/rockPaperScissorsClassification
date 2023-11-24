import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model("model")

classNames = ["none", "paper", "rock", "scissors"]


while True:
    success, img = cap.read()
    x, y, c = img.shape
    img = cv2.flip(img, 1)
    if not success:
        break
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_handedness and results.multi_hand_world_landmarks:
        landmarks = [results.multi_handedness[0].classification[0].index]
        for hand_landmark in results.multi_hand_world_landmarks:
            for lm in hand_landmark.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append(lmx)
                landmarks.append(lmy)
            mpDraw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)
        prediction = model.predict([landmarks])
        classID = np.argmax(prediction)
        className = classNames[classID]

        # show the prediction on the frame
        cv2.putText(img, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
