import time
import cv2
import torch
from ultralytics import YOLO
import supervision as sv
print('sv.__version__', sv.__version__)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main(model_path, video_path, device, camera_status):
    print("yolov8 info -", model_path)
    print("video input -", video_path)
    print("device info -", device)
    print("camera stat -", camera_status)

    # ---------------- zone definition ---------------- #
    # start and end position (1280, 720)
    if camera_status == 'xxl':
        LINE_START = sv.Point(700, 0)
        LINE_END = sv.Point(700, 1080)
    else:
        if camera_status:
            LINE_START = sv.Point(480, 0)
            LINE_END = sv.Point(480, 720)
        else:
            LINE_START = sv.Point(420, 480)
            LINE_END = sv.Point(420, 0)
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    # ---------------- zone definition ---------------- #

    # ---------------- loading model ---------------- #
    model = YOLO(model_path, task = 'detect')
    # model.to(device)
    # ---------------- loading model ---------------- #

    # ---------------- reading video ---------------- #
    url = video_path
    cap = cv2.VideoCapture(url)
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("stream  fps -", frame_fps)
    # ---------------- reading video ---------------- #

    # ---------------- count number ---------------- #
    in_count  = line_counter.in_count
    out_count = line_counter.out_count
    frame_length = 0
    # imgsz=(96,128)
    for result in model.track(url, imgsz = 128, conf = 0.4, device = device,
                              half = True, stream = True, persist = True, 
                              show = False, verbose = False, tracker = "./bytetrack_stream.yaml"):
        
        if frame_length == 0:
            start_time = time.time()
        elif frame_length == 1800:
            break
        frame_length = frame_length + 1

        if result.boxes.id is None:
            continue
        else:
            detections = sv.Detections.from_yolov8(result)
            detections = detections[detections.class_id == 0]
            detections.tracker_id = result.boxes.id.numpy().astype(int)
            line_counter.trigger(detections = detections)

            if in_count != line_counter.in_count:
                in_count = line_counter.in_count
                print('count numbers in ', line_counter.in_count)
            elif out_count != line_counter.out_count:
                out_count = line_counter.out_count
                print('count numbers out', line_counter.out_count)
        
    end_time = time.time()
    total_time = end_time - start_time
    print('time passed      ', total_time)
    print('time on one frame', total_time/frame_length)
    print('processing fps   ', frame_length/total_time)
    print('total numbers in ', line_counter.in_count)
    print('total numbers out', line_counter.out_count)
    cap.release()
    # ---------------- count number ---------------- #

if __name__ == '__main__':
    if torch.cuda.is_available():
        model_path = './weights/bestv8img128all.engine'
        device = torch.device(0)
    else:
        model_path = './weights/bestv8img128all.pt'
        device = 'cpu'
    
    video_path = './video/20230816162760_bbot00034.mp4'
    camera_status = False

    main(model_path, video_path, device, camera_status)