import cv2
import mediapipe as mp

# Initialize the MediaPipe Hands model
mp_hands = mp.solutions.hands

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the drawing utilities
mp_drawing = mp.solutions.drawing_utils 

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:

    while True:
        # Read the frame from the video capture
        ret, frame = cap.read()
        
        e1 = cv2.getTickCount()

        # Convert the frame to RGB for input to the model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform hand tracking on the frame
        results = hands.process(frame_rgb)

        # If hands are detected, draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get landmarks for thumb and pointer finger
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                # Calculate distance between thumb and pointer finger landmarks
                distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5

                # Set a threshold for pinch detection
                pinch_threshold = 0.02

                # If the distance is below the threshold, consider it as a pinch
                if distance < pinch_threshold:
                    cv2.putText(frame, "Pinch Detected", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        # stop the tick counter for computing the processing time for each frame
        e2 = cv2.getTickCount()
        # processing time in milliseconds
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
        
        cv2.putText(frame, "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + 'ms', (10, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)

        # Show the frame with landmarks drawn
        cv2.imshow('Hand Pose Estimation', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
