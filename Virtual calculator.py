import cv2
import numpy as np
import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Calculator logic
class Calculator:
    def __init__(self):
        self.current_input = ""
        self.result = ""

    def evaluate(self):
        try:
            self.result = str(eval(self.current_input))
        except Exception as e:
            self.result = "Error"
        return self.result

    def clear(self):
        self.current_input = ""
        self.result = ""

    def add_to_input(self, value):
        self.current_input += value

    def get_input(self):
        return self.current_input

    def get_result(self):
        return self.result

# Virtual Keypad with Simplified Hand Gestures
class VirtualKeypad:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # Width
        self.cap.set(4, 720)   # Height
        self.calculator = Calculator()

        # Define keypad buttons
        self.buttons = [
            ('7', (50, 100)), ('8', (250, 100)), ('9', (450, 100)), ('/', (650, 100)),
            ('4', (50, 250)), ('5', (250, 250)), ('6', (450, 250)), ('*', (650, 250)),
            ('1', (50, 400)), ('2', (250, 400)), ('3', (450, 400)), ('-', (650, 400)),
            ('0', (50, 550)), ('.', (250, 550)), ('=', (450, 550)), ('+', (650, 550)),
            ('C', (850, 100)), ('√', (850, 250)), ('^', (850, 400)), ('π', (850, 550))
        ]

        self.selected_button = None

    def draw_buttons(self, frame):
        for (text, pos) in self.buttons:
            x, y = pos
            color = (0, 255, 0) if self.selected_button == text else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + 180, y + 80), color, 2)
            cv2.putText(frame, text, (x + 50, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def detect_button_hover(self, x, y):
        for (text, pos) in self.buttons:
            bx, by = pos
            if bx <= x <= bx + 180 and by <= y <= by + 80:
                return text
        return None

    def is_fist_closed(self, landmarks):
        # Check if the fist is closed based on the distance between the thumb tip and index finger tip
        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
        return distance < 0.05  # Threshold for closed fist

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Hands
            results = hands.process(rgb_frame)

            # Draw buttons on the frame
            self.draw_buttons(frame)

            # Display current input and result
            cv2.putText(frame, f"Input: {self.calculator.get_input()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Result: {self.calculator.get_result()}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Detect hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get the coordinates of the palm center
                    palm_center = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    h, w, _ = frame.shape
                    x, y = int(palm_center.x * w), int(palm_center.y * h)

                    # Draw a circle at the palm center
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

                    # Detect button hover
                    self.selected_button = self.detect_button_hover(x, y)

                    # Detect fist closure for button press
                    if self.is_fist_closed(hand_landmarks):
                        if self.selected_button:
                            if self.selected_button == "=":
                                self.calculator.evaluate()
                            elif self.selected_button == "C":
                                self.calculator.clear()
                            elif self.selected_button == "√":
                                self.calculator.add_to_input("math.sqrt(")
                            elif self.selected_button == "^":
                                self.calculator.add_to_input("**")
                            elif self.selected_button == "π":
                                self.calculator.add_to_input(str(math.pi))
                            else:
                                self.calculator.add_to_input(self.selected_button)
                            self.selected_button = None  # Reset after press

            # Show the frame
            cv2.imshow("Virtual Calculator with Simplified Hand Gestures", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    keypad = VirtualKeypad()
    keypad.run()
