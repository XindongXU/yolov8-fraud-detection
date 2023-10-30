import time
import cv2
import torch
from ultralytics import YOLO
import supervision as sv
print('sv.__version__', sv.__version__)
from supervision.draw.color import Color

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from datetime import datetime

def main(model_path, video_path, device, camera_status):
    print('model used ', model_path)
    print('video input', video_path)
    print('device info', device)
    print('camera stat', camera_status)

    # ---------------- zone definition ---------------- #
    # start and end position (1280, 720), pour XXL (1920, 1080), pour 305 (320, 240)
    if camera_status == 'xxl':
        LINE_START = sv.Point(700, 0)
        LINE_END = sv.Point(700, 1080)
    else:
        if camera_status:
            LINE_START = sv.Point(480, 0)
            LINE_END = sv.Point(480, 720)
        else:
            # pour 34 (640, 480)
            LINE_START = sv.Point(420, 480)
            LINE_END = sv.Point(420, 0)

    line_counter = sv.LineZone(start = LINE_START, end = LINE_END)
    line_color = Color(r = 224, g = 57, b = 151)
    line_annotator = sv.LineZoneAnnotator(thickness = 2, text_thickness = 2, text_scale = 1, 
                                          color = line_color, text_offset = 10)
    box_annotator = sv.BoxAnnotator(thickness = 2, text_thickness = 1, text_scale = 0.5)

    url = video_path
    cap = cv2.VideoCapture(url)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(frame_fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    filename = './demo/demo_' + datetime.now().strftime("%Y%m%d%H%M%S")  + '.mp4'
    out = cv2.VideoWriter(filename, fourcc = fourcc,
                          fps = frame_fps, frameSize = (frame_width, frame_height))
    # ---------------- zone definition ---------------- #

    model = YOLO(model_path, task = 'detect')
    # model.to(device)
    names = {0: 'btl', 1: 'hand'}

    in_count  = line_counter.in_count
    out_count = line_counter.out_count
    frame_length = 0
    for result in model.track(url, imgsz = 128, conf = 0.4, device = device,
                              half = True, stream = True, persist = True, 
                              show = False, verbose = False, tracker = "./bytetrack_stream.yaml"):
                
        if frame_length == 0:
            start_time = time.time()
        # elif frame_length == 1800:
        #     break
        frame_length = frame_length + 1
        
        frame = result.orig_img
        if result.boxes.id is None:
            pass
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
            
            labels = [f"#{tracker_id} {names[class_id]} {confidence:0.2f}"
                                for _, _, confidence, class_id, tracker_id in detections]
            # plotting target detection visualization results
            frame = box_annotator.annotate(scene = frame, detections = detections, labels = labels)

        line_annotator.annotate(frame = frame, line_counter = line_counter)
        out.write(frame)        
            
    end_time = time.time()
    total_time = end_time - start_time
    print('time passed', total_time)
    print('time on one frame', total_time/frame_length)
    print('processing fps   ', frame_length/total_time)
    out.release()
    print('video is saved as', filename)
    cap.release()
    print('count numbers in ', line_counter.in_count)
    print('count numbers out', line_counter.out_count)
    # ---------------- count number ---------------- #

if __name__ == '__main__':
    model_path = './weights/bestv8img128all.engine'
    # video_path = 'rtsp://192.168.80.12:8554/entrance'
    video_path = 'rtsp://localhost:8554/entrance'
    video_path = './video/202308161627_bbot34.mp4'
    device = torch.device(0)
    camera_status = False

    main(model_path, video_path, device, camera_status)