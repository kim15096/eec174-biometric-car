import time
import cv2
import pygame
import threading
import Face_Detection
import Hand_Detection

# Initialize face and hand detection classes 
face_detection = Face_Detection.Face_Detection_Class()
hand_detection = Hand_Detection.Hand_Detection_Class()

# Global variables for face detection
roll_offset = 0
pitch_offset = 0
yaw_offset = 0
exit_thread = False
asleep = False
distracted = False

def main():
    hand_detector = hand_detection.setup_hand_detector()

    # Try the cv2 Optimization
    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)
        except:
            print("Optimization not applied")
            
    # Capture Video Input 
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    if not cap.isOpened(): 
        print("Cannot open camera")
        exit()

    ################# FOR PINCH DETECTION (CALIBRATION): #####################################################################################

    i = 0
    proc_time_total = 0
    is_pinch_detected = False
    is_face_valid = False

    while True:  
        e1 = cv2.getTickCount()
        t_now = time.perf_counter()

        ret, frame = cap.read()
        
        if not ret: 
            print("Can't receive frame from camera/stream end")
            exit()
        
        # flip the frame so it looks like a mirror.
        frame = cv2.flip(frame, 2)
        
        # display calibration text
        cv2.putText(frame, "CALIBRATING... PLEASE FACE FORWARD AND PINCH TO CALIBRATE", (30, 60), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 0), 2)
        
        # detect for hands
        hand_detection.run_hand_detection(frame, hand_detector)
        # get detection result
        detection_result = hand_detection.get_hand_detection_result()
        
        # if hand is detected
        if detection_result is not None:
            # check for pinch
            is_pinch_detected, calib_frame = hand_detection.check_for_pinch(detection_result, frame)
            # if pich detected, run the face detection for one frame
            if is_pinch_detected:
                global roll_offset, pitch_offset, yaw_offset
                # Get head offset values and store in global variables
                roll_offset, pitch_offset, yaw_offset = face_detection.run_detecton_calib(calib_frame)
                # Check if face is found when pinch is detected
                if roll_offset and pitch_offset and yaw_offset:
                    is_face_valid = True
                else:
                    is_face_valid = False
                    cv2.putText(frame, "FACE NOT FOUND... CALIBRATION UNSUCCESSFUL", (10, 250), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
                
        # stop the tick counter for computing the processing time for each frame
        e2 = cv2.getTickCount()
        
        # processing time in milliseconds
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
        proc_time_total += proc_time_frame_ms
        average_proc_time = proc_time_total / (i+1)
        
        # display proc time and avg proc time
        cv2.putText(frame, "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + 'ms', (10, 430), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        cv2.putText(frame, "AVG PROC. TIME FRAME:" + str(round(average_proc_time, 0)) + 'ms', (10, 500), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        # display frame on screen
        cv2.imshow("Calibrating", frame)
        
        # exit calibration if pinch detected and face is found
        if is_pinch_detected and is_face_valid:
            print("PINCH DETECTED")
            break
    
        if cv2.waitKey(20) & 0xFF == ord('q'):
            exit(0)
        
        i += 1
        
    ##### END OF PINCH CALIBRATION   ########################################################################################
    
    # Remove calibration window
    cv2.destroyAllWindows()

    ########### MAIN SCRIPT - POST CALIBRATION #############################################################
    
    # Create thread for warning sound to be run in the background
    sound_thread = threading.Thread(target=play_sound)
    sound_thread.start()
    
    i = 0
    proc_time_total = 0

    while True:
        t_now = time.perf_counter()
        e1 = cv2.getTickCount()

        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

        # if the frame comes from webcam, flip it so it looks like a mirror.
        frame = cv2.flip(frame, 2)

        # process frame to find face
        result, frame_size, gray_frame = face_detection.find_face(frame)

        # if at least one face is found
        if result:
            # getting face landmarks and then take only the bounding box of the biggest face
            landmarks = face_detection._get_landmarks(result)

            ear, gaze, frame_det, roll, pitch, yaw = face_detection.process_face(landmarks, frame, frame_size, gray_frame)
            
            # if the head pose estimation is successful, show the results
            if frame_det is not None:
                frame = frame_det
                
            # Recalculate roll, pitch and yaw given offset values from calibration
            roll -= roll_offset
            pitch -= pitch_offset
            yaw -= yaw_offset
    
            global asleep
            global distracted
            # Evaluate scores given the offset values we found in calibration
            asleep, distracted =face_detection.evaluate_scores(t_now, ear, gaze, roll, pitch, yaw)

            # show the real-time EAR score
            if ear is not None:
                cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

            # show the real-time Gaze Score
            if gaze is not None:
                cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
                
            # display head position values
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
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
            if distracted:
                cv2.putText(frame, "DISTRACTED!", (10, 340),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
        
        # If there is no face detected
        else:
            cv2.putText(frame, "FACE NOT FOUND!", (10, 340),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
        e2 = cv2.getTickCount()
        
        # processign time in milliseconds
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
        proc_time_total += proc_time_frame_ms
        avg_proc_time = proc_time_total / (i+1)
        
        # Show proc time and avg proc time
        cv2.putText(frame, "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + 'ms', (10, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)
        cv2.putText(frame, "AVG PROC. TIME FRAME:" + str(round(avg_proc_time, 0)) + 'ms', (10, 500), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)

        # show the frame on screen
        cv2.imshow("Press 'q' to terminate", frame)

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord('q'):
            # raise exit_thread flag to exit from thread
            global exit_thread
            exit_thread = True
            time.sleep(1)
            sound_thread.join()
            break
        
        i += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    ################### END OF FACE DETECTION #########################################################

    return

# THREAD: PLAY WARNING SOUND IN THE BACKGROUND
def play_sound():
    while True:
        if exit_thread:
            break         

        if asleep:
            sound_file = 'driver_state_detection/assets/warning_wake_up.mp3'   
            pygame.mixer.init()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        
        if distracted:
            sound_file = 'driver_state_detection/assets/warning_sound.mp3'   
            pygame.mixer.init()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        
        time.sleep(0.1)
        
if __name__ == "__main__":
    main()