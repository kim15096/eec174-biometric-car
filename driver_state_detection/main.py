import time
import argparse
import sys

import cv2
import numpy as np
import mediapipe as mp
import os
import pygame
import random
import threading

from face_modules.Eye_Dector_Module import EyeDetector as EyeDet
from face_modules.Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from face_modules.Attention_Scorer_Module import AttentionScorer as AttScorer

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

probability_minimum = 0.5
threshold = 0.3

# For Pinch Detection (Calibration)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables to calculate FPS
START_TIME = time.time()
DETECTION_RESULT = None
roll_offset = 0
pitch_offset = 0
yaw_offset = 0
exit_thread = False
asleep = False

def play_sound():
    while True:
        if exit_thread:
            break
        sound_file = 'driver_state_detection/assets/warning_wake_up.mp3'            

        if asleep:
            print("PLAYING WAKE UP SOUND", random.randint(1, 10))
            pygame.mixer.init()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        
        time.sleep(0.1)

def save_result(result: vision.HandLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global DETECTION_RESULT
        DETECTION_RESULT = result
        
# camera matrix obtained from the camera calibration script, using a 9x6 chessboard
camera_matrix = np.array(
    [[899.12150372, 0., 644.26261492],
     [0., 899.45280671, 372.28009436],
     [0, 0,  1]], dtype="double")

# distortion coefficients obtained from the camera calibration script, using a 9x6 chessboard
dist_coeffs = np.array(
    [[-0.03792548, 0.09233237, 0.00419088, 0.00317323, -0.15804257]], dtype="double")

def _get_landmarks(lms):
    if not lms:
        return None

    landmarks = [np.array([point.x, point.y, point.z]) for point in lms[0].landmark]

    landmarks = np.array(landmarks)

    landmarks[landmarks[:, 0] < 0., 0] = 0.
    landmarks[landmarks[:, 0] > 1., 0] = 1.
    landmarks[landmarks[:, 1] < 0., 1] = 0.
    landmarks[landmarks[:, 1] > 1., 1] = 1.

    return landmarks


def main():
    # Setup code (hide set up code)
    if True:
        # Initialize the hand landmarker model
        # Get the absolute path to the directory containing the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path to the model asset relative to the script directory
        model_asset_path = os.path.join(script_dir, 'hand_modules')
        model_asset_path = os.path.join(model_asset_path, 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=save_result)
        detector = vision.HandLandmarker.create_from_options(options)

        parser = argparse.ArgumentParser(description='Driver State Detection')

        # selection the camera number, default is 0 (webcam)
        parser.add_argument('-c', '--camera', type=int,
                            default=0, metavar='', help='Camera number, default is 0 (webcam)')

        # visualisation parameters
        parser.add_argument('--show_proc_time', type=bool, default=True,
                            metavar='', help='Show the processing time for a single frame, default is true')
        parser.add_argument('--show_eye_proc', type=bool, default=False,
                            metavar='', help='Show the eyes processing, deafult is false')
        parser.add_argument('--show_axis', type=bool, default=True,
                            metavar='', help='Show the head pose axis, default is true')
        parser.add_argument('--verbose', type=bool, default=False,
                            metavar='', help='Prints additional info, default is false')

        # Attention Scorer parameters (EAR, Gaze Score, Pose)
        parser.add_argument('--smooth_factor', type=float, default=0.5,
                            metavar='', help='Sets the smooth factor for the head pose estimation keypoint smoothing, default is 0.5')
        parser.add_argument('--ear_thresh', type=float, default=0.25,
                            metavar='', help='Sets the EAR threshold for the Attention Scorer, default is 0.15')
        parser.add_argument('--ear_time_thresh', type=float, default=2,
                            metavar='', help='Sets the EAR time (seconds) threshold for the Attention Scorer, default is 2 seconds')
        parser.add_argument('--gaze_thresh', type=float, default=0.015,
                            metavar='', help='Sets the Gaze Score threshold for the Attention Scorer, default is 0.2')
        parser.add_argument('--gaze_time_thresh', type=float, default=2, metavar='',
                            help='Sets the Gaze Score time (seconds) threshold for the Attention Scorer, default is 2. seconds')
        parser.add_argument('--pitch_thresh', type=float, default=20,
                            metavar='', help='Sets the PITCH threshold (degrees) for the Attention Scorer, default is 30 degrees')
        parser.add_argument('--yaw_thresh', type=float, default=20,
                            metavar='', help='Sets the YAW threshold (degrees) for the Attention Scorer, default is 20 degrees')
        parser.add_argument('--roll_thresh', type=float, default=20,
                            metavar='', help='Sets the ROLL threshold (degrees) for the Attention Scorer, default is 30 degrees')
        parser.add_argument('--pose_time_thresh', type=float, default=2.5,
                            metavar='', help='Sets the Pose time threshold (seconds) for the Attention Scorer, default is 2.5 seconds')

        # parse the arguments and store them in the args variable dictionary
        args = parser.parse_args()

        if args.verbose:
            print(f"Arguments and Parameters used:\n{args}\n")

        if not cv2.useOptimized():
            try:
                cv2.setUseOptimized(True)  # set OpenCV optimization to True
            except:
                print(
                    "OpenCV optimization could not be set to True, the script may be slower than expected")
                

        detector_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5,
                                                refine_landmarks=True, max_num_faces=1 )

        # instantiation of the eye detector and pose estimator objects
        Eye_det = EyeDet(show_processing=args.show_eye_proc)

        Head_pose = HeadPoseEst(show_axis=args.show_axis)

        # instantiation of the attention scorer object, with the various thresholds
        # NOTE: set verbose to True for additional printed information about the scores
        t0 = time.perf_counter()
        Scorer = AttScorer(t_now=t0, ear_thresh=args.ear_thresh, gaze_time_thresh=args.gaze_time_thresh,
                        roll_thresh=args.roll_thresh, pitch_thresh=args.pitch_thresh,
                        yaw_thresh=args.yaw_thresh, ear_time_thresh=args.ear_time_thresh,
                        gaze_thresh=args.gaze_thresh, pose_time_thresh=args.pose_time_thresh,
                        verbose=args.verbose)

    # capture the input from the default system camera (camera number 0)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()

    i = 0
    
    userPinched = False

    while True:  # infinite loop for calibrating
    
        e1 = cv2.getTickCount()
        t_now = time.perf_counter()

        ret, frame = cap.read()  # read a frame from the webcam
        
        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break
        
        width  = cap.get(3)  
        height = cap.get(4)
        
         # if the frame comes from webcam, flip it so it looks like a mirror.
        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        ## FOR PINCH DETECTION: #####################################################################################

        cv2.putText(frame, "CALIBRATING... PLEASE FACE FORWARD AND PINCH TO CALIBRATE", (30, 60), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 0), 2)
        
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run hand landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        current_frame = frame

        if DETECTION_RESULT:
            # Draw landmarks and indicate handedness.
            for idx in range(len(DETECTION_RESULT.hand_landmarks)):
                hand_landmarks = DETECTION_RESULT.hand_landmarks[idx]
                handedness = DETECTION_RESULT.handedness[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                    z=landmark.z) for landmark
                    in hand_landmarks
                ])
                mp_drawing.draw_landmarks(
                    current_frame,
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
                    cv2.putText(current_frame, "Pinch Detected", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    userPinched = True
                    runFaceDetection(current_frame)
                        # stop the tick counter for computing the processing time for each frame
        e2 = cv2.getTickCount()
                    # processing time in milliseconds
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
        cv2.putText(current_frame, "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + 'ms', (10, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 0, 255), 1)

        ########################################################################################

        def runFaceDetection(current_frame):
            global roll_offset, pitch_offset, yaw_offset
        # # start the tick counter for computing the processing time for each frame
            e1 = cv2.getTickCount()

            # transform the BGR frame in grayscale
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            # get the frame size
            curr_frame_size = current_frame.shape[1], current_frame.shape[0]

            # apply a bilateral filter to lower noise but keep frame details. create a 3D matrix from gray image to give it to the model
            gray = np.expand_dims(cv2.bilateralFilter(gray, 5, 10, 10), axis=2)
            gray = np.concatenate([gray, gray, gray], axis=2)

            # find the faces using the face mesh model
            lms = detector_face.process(gray).multi_face_landmarks

            if lms:  # process the frame only if at least a face is found
                # getting face landmarks and then take only the bounding box of the biggest face
                landmarks = _get_landmarks(lms)

                # shows the eye keypoints (can be commented)
                Eye_det.show_eye_keypoints(
                    color_frame=current_frame, landmarks=landmarks, frame_size=curr_frame_size)


                # compute the head pose
                frame_det, roll, pitch, yaw = Head_pose.get_pose(
                    frame=current_frame, landmarks=landmarks, frame_size=curr_frame_size)
            
                roll_offset = roll
                pitch_offset = pitch
                yaw_offset = yaw

        # # show the frame on screen
        cv2.imshow("Calibrating", frame)
        if userPinched or cv2.waitKey(20) & 0xFF == ord('q'):
            break
        
        i += 1
        
    cv2.destroyAllWindows()

    ## MAIN SCRIPT - POST CALIBRATION
    sound_thread = threading.Thread(target=play_sound)
    sound_thread.start()
    
    while True:  # infinite loop for webcam video capture
        t_now = time.perf_counter()

        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

         # if the frame comes from webcam, flip it so it looks like a mirror.
        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        # start the tick counter for computing the processing time for each frame
        e1 = cv2.getTickCount()

        # transform the BGR frame in grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get the frame size
        frame_size = frame.shape[1], frame.shape[0]

        # apply a bilateral filter to lower noise but keep frame details. create a 3D matrix from gray image to give it to the model
        gray = np.expand_dims(cv2.bilateralFilter(gray, 5, 10, 10), axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        # find the faces using the face mesh model
        lms = detector_face.process(gray).multi_face_landmarks

        if lms:  # process the frame only if at least a face is found
            # getting face landmarks and then take only the bounding box of the biggest face
            landmarks = _get_landmarks(lms)

            # shows the eye keypoints (can be commented)
            Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size)

            # compute the EAR score of the eyes
            ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

            # compute the Gaze Score
            gaze = Eye_det.get_Gaze_Score(
                frame=gray, landmarks=landmarks, frame_size=frame_size)

            # compute the head pose
            frame_det, roll, pitch, yaw = Head_pose.get_pose(
                frame=frame, landmarks=landmarks, frame_size=frame_size)
            
            roll -= roll_offset
            pitch -= pitch_offset
            yaw -= yaw_offset
            
             # evaluate the scores for EAR, GAZE and HEAD POSE
            global asleep
            
            asleep, looking_away, distracted = Scorer.eval_scores(t_now=t_now,
                                                                  ear_score=ear,
                                                                  gaze_score=gaze,
                                                                  head_roll=roll,
                                                                  head_pitch=pitch,
                                                                  head_yaw=yaw)

            # if the head pose estimation is successful, show the results
            if frame_det is not None:
                frame = frame_det

            # show the real-time EAR score
            if ear is not None:
                cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

            # show the real-time Gaze Score
            if gaze is not None:
                cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
            
            if roll is not None:
                cv2.putText(frame, "roll:"+str(roll.round(1)[0]), (450, 40),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
            if pitch is not None:
                cv2.putText(frame, "pitch:"+str(pitch.round(1)[0]), (450, 70),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
            if yaw is not None:
                cv2.putText(frame, "yaw:"+str(yaw.round(1)[0]), (450, 100),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)

            # if the state of attention of the driver is not normal, show an alert on screen
                        
            if asleep:
                cv2.putText(frame, "ASLEEP!", (10, 300),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                         
            if looking_away:
                cv2.putText(frame, "LOOKING AWAY!", (10, 320),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            if distracted:
                cv2.putText(frame, "DISTRACTED!", (10, 340),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
        else:
            cv2.putText(frame, "FACE NOT FOUND!", (10, 340),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            

        # stop the tick counter for computing the processing time for each frame
        e2 = cv2.getTickCount()
        # processign time in milliseconds
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000

        if args.show_proc_time:
            cv2.putText(frame, "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + 'ms', (10, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)

        # show the frame on screen
        cv2.imshow("Press 'q' to terminate", frame)

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord('q'):
            global exit_thread
            exit_thread = True
            time.sleep(1)
            sound_thread.join()
            break
        
        i += 1
    
    
    cap.release()
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()