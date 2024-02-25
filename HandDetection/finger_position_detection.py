import cv2
import mediapipe as mp
import math

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)


def calculate_finger_openness(finger_landmarks):
    """Calculate openness percentage of a finger"""
    base_length = math.sqrt((finger_landmarks[0][0] - finger_landmarks[1][0])**2 + (
        finger_landmarks[0][1] - finger_landmarks[1][1])**2)
    finger_length = math.sqrt((finger_landmarks[1][0] - finger_landmarks[2][0])**2 + (
        finger_landmarks[1][1] - finger_landmarks[2][1])**2)
    openness = (finger_length / base_length) * 100
    return openness


while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = []
            for lm in handslms.landmark:
                # Get the landmark coordinates
                h, w, c = frame.shape
                lmx = int(lm.x * w)
                lmy = int(lm.y * h)
                landmarks.append([lmx, lmy])

            # Calculate finger openness
            fingers = {
                "Thumb": calculate_finger_openness([landmarks[i] for i in [2, 3, 4]]),
                "Index": calculate_finger_openness([landmarks[i] for i in [5, 6, 8]]),
                "Middle": calculate_finger_openness([landmarks[i] for i in [9, 10, 12]]),
                "Ring": calculate_finger_openness([landmarks[i] for i in [13, 14, 16]]),
                "Pinky": calculate_finger_openness([landmarks[i] for i in [17, 18, 20]])
            }

            # Draw lines to represent finger openness
            for i, (finger, openness) in enumerate(fingers.items()):
                # Calculate text position
                text_x = w - 150  # Horizontal position
                text_y = 20 + i * 40  # Vertical position

                # Write percentage
                cv2.putText(frame, f'{finger}: {int(openness)}%', (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw lines between finger joints
                for j in range(1, 4):
                    cv2.line(frame, tuple(
                        landmarks[i*4+j-1]), tuple(landmarks[i*4+j]), (255, 0, 0), 2)

    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
