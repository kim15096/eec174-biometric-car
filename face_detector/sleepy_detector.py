import dlib
import cv2
import numpy as np
import time

head_not_forward_count = 0
nod_count = 0
blink_count = 0
yawn_count = 0
face_detected = False
one_sec = 30
eye_closed_frame = 0
start_time = time.time()
frame_count = 0
yawn_frame = 0

#work cited: TianjingWu https://github.com/TianxingWu for detecting faces

def landmarks_to_np(landmarks, dtype="int"): 
    num = landmarks.num_parts
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

predictor_path = "./data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0) # connect to your camera
queue = np.zeros(30,dtype=int)
queue = queue.tolist()
frames_since_last_blink = 0
frames_since_last_yawn = 0


while(cap.isOpened()):
    
    face_detected = False
    blink_detected = False
    frames_since_last_blink += 1
    frames_since_last_yawn += 1
    frame_count += 1
    
    # read frame
    _, img = cap.read()
    e1 = cv2.getTickCount()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    cv2.putText(img, "Blink Count: {}".format(blink_count), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Yawn Count: {}".format(yawn_count), (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        fps = frame_count
        start_time = time.time()
        frame_count = 0
        
    cv2.putText(img, f"FPS: {int(fps)}", (img.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # for every face
    for i, rect in enumerate(rects):
        face_detected = True
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        
        # draw boxes
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, "Face {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # get landmarks        
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)
        
            # Draw rectangles around eyes
        left_eye_box = landmarks[36:42].astype(int)
        right_eye_box = landmarks[42:48].astype(int)

        cv2.polylines(img, [left_eye_box], isClosed=True, color=(255, 255, 255), thickness=1)
        cv2.polylines(img, [right_eye_box], isClosed=True, color=(255, 255, 255), thickness=1)

     
        # eye distance
        d1 =  np.linalg.norm(landmarks[37]-landmarks[41])
        d2 =  np.linalg.norm(landmarks[38]-landmarks[40])
        d3 =  np.linalg.norm(landmarks[43]-landmarks[47])
        d4 =  np.linalg.norm(landmarks[44]-landmarks[46])
        d_mean = (d1+d2+d3+d4)/4
        d5 =np.linalg.norm(landmarks[36]-landmarks[39])
        d6 =np.linalg.norm(landmarks[42]-landmarks[45])
        d_reference = (d5+d6)/2
        d_judge = d_mean/d_reference
        
        if d_judge<0.2 and frames_since_last_blink>one_sec/4:
            blink_count += 1
            frames_since_last_blink = 0
        
        flag = int(d_judge<0.2) #0.2 is the threshod, adjust bigger if you have larger eyes unlike me

        queue = queue[1:len(queue)] + [flag] #enqueue flag

        
        # Count blinks
        if sum(queue) > len(queue)/2:
            cv2.putText(img, "Eyes Closed!", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    
        mouth_open_threshold = 10  # Adjust the threshold as needed
        if landmarks[66][1] - landmarks[62][1] > mouth_open_threshold:
            yawn_frame += 1
            if yawn_frame >= 10 and frames_since_last_blink>one_sec*3: #if surpass 10 frames and more than one secend
                yawn_count += 1
                frames_since_last_yawn = 0
                yawn_frame = 0
            

        if sum(queue) <= len(queue)/2:
            cv2.putText(img, "SAFE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
    if not face_detected:
        cv2.putText(img, "No Face Detected!", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    # stop the tick counter for computing the processing time for each frame
    e2 = cv2.getTickCount()
    # processign time in milliseconds
    proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
    cv2.putText(img, "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + 'ms', (10, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)
        
    cv2.imshow("Camera", img)
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):  # 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
