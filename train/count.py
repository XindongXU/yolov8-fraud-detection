import time
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
from ultralytics import YOLO
import torch

import supervision as sv
from supervision.draw.color import Color
from img2autohist import hisEqulColor

# ---------------- zone definition ---------------- #
# start and end position (1280, 720)
LINE_START = sv.Point(300, 0)
LINE_END = sv.Point(300, 720)
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

# visualisation for the line
line_color = Color(r=224, g=57, b=151)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1, color=line_color, text_offset=10)

# visualisation for target detection
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5)
# ---------------- zone definition ---------------- #

# ---------------- import model v8 ---------------- #
# model = YOLO('./runs/detect/train_202307120907_v5_n_s_dataset6/weights/best.pt')
model = torch.load("./runs/detect/train_202307120907_v5_n_s_dataset6/weights/best.pt")

# weights = "./runs/detect/train_202307120907_v5_n_s_dataset6/weights/best.pt"
# model = torch.hub.load('ultralytics/yolov5', 'custom', weights, device=0)
# ---------------- import model v8 ---------------- #

# ---------------- video in/out ---------------- #
test_file = "./video/202306161506_bbot00007.mp4"
# test_file = "./video/202306161533_bbot00016.mp4"
# video for testing the script, so we choose a video from training dataset
cap = cv2.VideoCapture(test_file)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print('frame_length', frame_length)

# creation of VideoWriter, set the output file name and encoding format, as well as the frame rate and frame size
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S")
filename = filename + '_bbot000' + test_file[28:30] + '.mp4'
out = cv2.VideoWriter('./demo_count/demo_' + filename, fourcc, frame_fps, (frame_width, frame_height))
# ---------------- video in/out ---------------- #

# ---------------- count number ---------------- #
start_time = time.time()
for i in tqdm(range(frame_length-1)):
    # print(i)
    ret, frame = cap.read()
    if not ret:
        break
    frame = hisEqulColor(frame, clip = 5, grid = 10)
    results = model.track(frame, imgsz=640, persist=True, show=False, verbose=False)
    
    if results[0].boxes.id is None:  # add this check
        continue

    detections = sv.Detections.from_yolov8(results[0])
    detections = detections[detections.class_id == 0]
    detections.tracker_id = results[0].boxes.id.numpy().astype(int)

    class_ids = detections.class_id     # class ID
    confidences = detections.confidence # confidence
    tracker_ids = detections.tracker_id # object ID
    labels = ['#{} {} {:.1f}'.format(tracker_ids[i], model.names[class_ids[i]], confidences[i]*100) for i in range(len(class_ids))]
    
    # plotting target detection visualization results
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    
    # line crossing detection
    line_counter.trigger(detections=detections)
    line_annotator.annotate(frame=frame, line_counter=line_counter)

    out.write(frame)
end_time = time.time()
total_time = end_time - start_time

cv2.destroyAllWindows()
out.release()
cap.release()
print('video is saved as', './demo_count/demo_' + filename)
print('count numbers in ', line_counter.in_count)
print('count numbers out', line_counter.out_count)
print('time on one frame', total_time/frame_length)
print('processing fps   ', frame_length/total_time)