import numpy as np
import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

weights = "tiny_best.pt"
video_path = ""
cap = cv2.VideoCapture(0)
device = 0 #If you want to run with cpu -> device = "cpu"
device = select_device(str(device))
model = attempt_load(weights, map_location=device)

imgsz = 640
half = True

conf_thres = 0.3
iou_thres = 0.3

if half:
    model.half()  # to FP16

names = model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

if device != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once for warmup


def getColor(x, y, w, h, frame):
    cropped_img = frame[y:h, x:w]
    cv2.imwrite("cropped_img.jpg", cropped_img)
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    top = max(gray_hist)
    for x, y in enumerate(gray_hist):
        if gray_hist[x] == top:
            index = x
            break
    if index < 150:
        return False
    else:
        return True

while True:
    _, im0 = cap.read()
    e1 = cv2.getTickCount()
    img = letterbox(im0, imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    #img = np.ascontiguousarray(img)
    img = torch.from_numpy(img.copy()).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None,
                                   agnostic=False)  # for filtering change classes
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            x1, y1, x2, y2 = det[:, :4].cpu().numpy()[0]
            for *xyxy, conf, cls in reversed(det):
                if getColor(int(x1), int(y1), int(x2), int(y2), im0):
                    cv2.putText(im0, "Screen is on", (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
                plot_one_box(xyxy, im0, color=colors[int(cls)], line_thickness=3)
    e2 = cv2.getTickCount()
    # processign time in milliseconds
    proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
    cv2.putText(im0, "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + 'ms', (10, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)

    cv2.imshow("Frame", im0)
    ch = cv2.waitKey(1)  # 1 millisecond
    if ch == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break
