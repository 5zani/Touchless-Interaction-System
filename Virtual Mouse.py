import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Get screen size for scaling coordinates
screen_width, screen_height = pyautogui.size()

# Set up the webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to avoid mirror effect
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract the coordinates of the index finger tip (landmark 8)
            x, y = int(hand_landmarks.landmark[8].x * frame_width), int(hand_landmarks.landmark[8].y * frame_height)

            # Convert the coordinates to screen size
            screen_x = np.interp(x, [0, frame_width], [0, screen_width])
            screen_y = np.interp(y, [0, frame_height], [0, screen_height])

            # Move the mouse cursor to the calculated position
            pyautogui.moveTo(screen_x, screen_y)

            # Logic for clicking can be added here
            # If landmark 12 (middle finger tip) is lower than landmark 8 (index finger tip), perform a click
            if hand_landmarks.landmark[12].y > hand_landmarks.landmark[8].y:
                pyautogui.click()

            # Draw landmarks on the frame (optional)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Virtual Mouse", frame)

    # Press 'q' to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
