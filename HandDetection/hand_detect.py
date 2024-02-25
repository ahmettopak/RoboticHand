import cv2
import mediapipe as mp

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

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
            for id, lm in enumerate(handslms.landmark):
                # Get the landmark coordinates
                h, w, c = frame.shape
                lmx = int(lm.x * w)
                lmy = int(lm.y * h)
                landmarks.append([lmx, lmy])

                # Draw landmarks on frames
                mpDraw.draw_landmarks(
                    frame, handslms, mpHands.HAND_CONNECTIONS)

            # Define finger names
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

            # Calculate finger openness as percentage
            finger_openness = []
            for i in range(5):
                finger_length = landmarks[i+1][1] - landmarks[i][1]
                finger_openness.append(int((finger_length / h) * 100))

            # Print finger names and openness percentage
            for i, finger in enumerate(finger_names):
                print(finger + ':', finger_openness[i], '%')

            # Display finger names and openness percentage on the frame
            for i, finger in enumerate(finger_names):
                cv2.putText(frame, f'{finger}: {finger_openness[i]}%', (10, 30+i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
