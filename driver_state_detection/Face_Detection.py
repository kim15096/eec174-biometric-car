import numpy as np
import cv2
import time
import mediapipe as mp
from face_modules.Eye_Dector_Module import EyeDetector as EyeDet
from face_modules.Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from face_modules.Attention_Scorer_Module import AttentionScorer as AttScorer

# instantiation of the eye detector and pose estimator objects
Eye_det = EyeDet()
Head_pose = HeadPoseEst(show_axis=True)

# Attention Scorer Parameters
smooth_factor = 0.5
ear_thresh = 0.25
ear_time_thresh = 2.5
gaze_thresh = 0.015
gaze_time_thresh = 2
pitch_thresh = 20
yaw_thresh = 20
roll_thresh = 20
pose_time_thresh = 2

# instantiation of the attention scorer object, with the various thresholds

t0 = time.perf_counter()

Scorer = AttScorer(t_now=t0, ear_thresh=ear_thresh, gaze_time_thresh=gaze_time_thresh,
                roll_thresh=roll_thresh, pitch_thresh=pitch_thresh,
                yaw_thresh=yaw_thresh, ear_time_thresh=ear_time_thresh,
                gaze_thresh=gaze_thresh, pose_time_thresh=pose_time_thresh)

detector_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                            min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5,
                                            refine_landmarks=True, max_num_faces=1 )

class Face_Detection_Class():
    def __init__(self) -> None:
        pass
    
    def _get_landmarks(self, lms):
        if not lms:
            return None

        landmarks = [np.array([point.x, point.y, point.z]) for point in lms[0].landmark]

        landmarks = np.array(landmarks)

        landmarks[landmarks[:, 0] < 0., 0] = 0.
        landmarks[landmarks[:, 0] > 1., 0] = 1.
        landmarks[landmarks[:, 1] < 0., 1] = 0.
        landmarks[landmarks[:, 1] > 1., 1] = 1.
        
        return landmarks
    
    def run_detecton_calib(self, frame):
        # transform the BGR frame in grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get the frame size
        curr_frame_size = frame.shape[1], frame.shape[0]

        # apply a bilateral filter to lower noise but keep frame details. create a 3D matrix from gray image to give it to the model
        gray = np.expand_dims(cv2.bilateralFilter(gray, 5, 10, 10), axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        # find the faces using the face mesh model
        lms = detector_face.process(gray).multi_face_landmarks

        if lms:  # process the frame only if at least a face is found
            # getting face landmarks and then take only the bounding box of the biggest face
            landmarks = self._get_landmarks(lms)

            # shows the eye keypoints (can be commented)
            Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=curr_frame_size)


            # compute the head pose
            frame_det, roll, pitch, yaw = Head_pose.get_pose(
                frame=frame, landmarks=landmarks, frame_size=curr_frame_size)
        
            roll_offset = roll
            pitch_offset = pitch
            yaw_offset = yaw

            return roll_offset, pitch_offset, yaw_offset
        
        else:
            return None, None, None
    
    def find_face(self, frame):
        # transform the BGR frame in grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # get the frame size
        frame_size = frame.shape[1], frame.shape[0]
        
        # apply a bilateral filter to lower noise but keep frame details. create a 3D matrix from gray image to give it to the model
        gray = np.expand_dims(cv2.bilateralFilter(gray, 5, 10, 10), axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)
        
        lms = detector_face.process(gray).multi_face_landmarks

        return lms, frame_size, gray
    
    def process_face(self, landmarks, frame, frame_size, gray_frame):
        # shows the eye keypoints (can be commented)
        Eye_det.show_eye_keypoints(
            color_frame=frame, landmarks=landmarks, frame_size=frame_size)
        
        # compute the EAR score of the eyes
        ear = Eye_det.get_EAR(frame=gray_frame, landmarks=landmarks)
        
        # compute the Gaze Score
        gaze = Eye_det.get_Gaze_Score(
            frame=gray_frame, landmarks=landmarks, frame_size=frame_size)
        
        # compute the head pose
        frame_det, roll, pitch, yaw = Head_pose.get_pose(
            frame=frame, landmarks=landmarks, frame_size=frame_size)
        
        return ear, gaze, frame_det, roll, pitch, yaw
    
    def evaluate_scores(self, t_now, ear, gaze, roll, pitch, yaw):
        asleep, distracted = Scorer.eval_scores(t_now=t_now, ear_score=ear, gaze_score=gaze, head_roll=roll, head_pitch=pitch, head_yaw=yaw)
        return asleep, distracted