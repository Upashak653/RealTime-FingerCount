import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define hand tracking
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Finger tip landmarks (thumb tip, index tip, etc.)
finger_tips = [4, 8, 12, 16, 20]

# Open the webcam
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Flip frame to avoid mirror effect and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)

    finger_count = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks in numpy array
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            h, w, _ = frame.shape
            landmarks *= [w, h]
            landmarks = landmarks.astype(int)

            # Thumb: Check if thumb tip is to the right or left of thumb MCP
            if landmarks[finger_tips[0]][0] > landmarks[finger_tips[0]-1][0]:
                finger_count += 1

            # Other fingers: Tip should be above the joint below it
            for tip in finger_tips[1:]:
                if landmarks[tip][1] < landmarks[tip - 2][1]:
                    finger_count += 1

    # Show finger count on screen
    cv2.putText(frame, f'Fingers: {finger_count}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow('Finger Counter', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
