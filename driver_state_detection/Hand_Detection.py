from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os
import cv2
import time
import mediapipe as mp

# For Pinch Detection (Calibration)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class Hand_Detection_Class():
    def __init__(self) -> None:
        self.DETECTION_RESULT = None
        pass
    
    def setup_hand_detector(self):
        # Get path to modules and model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_asset_path = os.path.join(script_dir, 'hand_modules')
        model_asset_path = os.path.join(model_asset_path, 'hand_landmarker.task')
        
        # Set options for the hand land marker
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self.hand_detection_result)
        
        # Create hand_detector with the options
        detector = vision.HandLandmarker.create_from_options(options)

        return detector
    
    def run_hand_detection(self, frame, hand_detector):
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run hand landmarker using the model.
        hand_detector.detect_async(mp_image, time.time_ns() // 1_000_000)

    def hand_detection_result(self, result: vision.HandLandmarkerResult, unused_output_image: mp.Image, timestamp_ms: int):
        self.DETECTION_RESULT = result
    
    def get_hand_detection_result(self):
        return self.DETECTION_RESULT
    
    def check_for_pinch(self, detection_result, frame):
        # Draw landmarks and indicate handedness.
        for idx in range(len(detection_result.hand_landmarks)):
            hand_landmarks = detection_result.hand_landmarks[idx]
            handedness = detection_result.handedness[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                z=landmark.z) for landmark
                in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Get landmarks for thumb and pointer finger
            thumb_tip = hand_landmarks_proto.landmark[4]
            index_tip = hand_landmarks_proto.landmark[8]

            # Calculate distance between thumb and pointer finger landmarks
            distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5

            # Set a threshold for pinch detection
            pinch_threshold = 0.05

            # If the distance is below the threshold, consider it as a pinch
            if distance < pinch_threshold:
                cv2.putText(frame, "Pinch Detected", (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                return True, frame
        
        return False, frame