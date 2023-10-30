import time
import cv2
import torch
from ultralytics import YOLO
import supervision as sv
from supervision.draw.color import Color
from datetime import datetime
import os
import sys
import logging
import subprocess as sp
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('appsrc ! videoconvert ! '
                      'x264enc noise-reduction=10000 speed-preset=ultrafast tune=zerolatency ! '
                      'rtph264pay config-interval=1 pt=96 !'
                      'tcpserversink host=0.0.0.0 port=5000 sync=false',
                      fourcc, 9.0, (320, 180))

# FFmpeg command to send video to the RTSP server
command = ['ffmpeg',
           '-f', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', '{}x{}'.format(320, 180),  # keep this the same as your actual frame size
           '-r', str(9),
           '-i', '-',
           '-an',
           '-vf', 'scale=320:180',  # this scales the video down to 320x180
           '-c:v', 'libx264',
           '-preset', 'ultrafast',
           '-tune', 'zerolatency',
           '-f', 'rtsp',
           '-rtsp_transport', 'udp',
           'rtsp://192.168.1.58:8554/stream']

proc = sp.Popen(command, stdin=sp.PIPE, shell=False)

def get_unique_filename(base_filename):
    """
    Returns a filename based on base_filename that does not exist yet.
    If base_filename exists, it appends (1), (2), etc. until a unique filename is found.
    """
    # Split filename and extension
    filename, extension = os.path.splitext(base_filename)

    counter = 1
    while os.path.exists(base_filename):
        base_filename = f"{filename}({counter}){extension}"
        counter += 1

    return base_filename

class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def main(model_path, video_path, device, camera_status, save_dir):
    
    logging.info("yolov8 info - %s", model_path)
    logging.info("video input - %s", video_path)
    logging.info("device info - %s", device)
    logging.info("camera stat - %s", camera_status)
    logging.info("demos saved - %s", save_dir)

    # ---------------- zone definition ---------------- #
    url = video_path
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {url}")
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
    logging.info("stream  fps - %.2f", frame_fps)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (320, 180))
        proc.stdin.write(frame.tobytes())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # start and end position (1280, 720), pour XXL (1920, 1080), pour 305 (320, 240)
    if camera_status == 'xxl':
        LINE_START = sv.Point(700, 0)
        LINE_END = sv.Point(700, 1080)
    else:
        if camera_status:
            LINE_START = sv.Point(240, 0)
            LINE_END = sv.Point(240, 480)
        else:
            # pour 34 (640, 480)
            LINE_START = sv.Point(420, 480)
            LINE_END = sv.Point(420, 0)

    line_counter = sv.LineZone(start = LINE_START, end = LINE_END)
    line_color = Color(r = 224, g = 57, b = 151)
    line_annotator = sv.LineZoneAnnotator(thickness = 2, text_thickness = 2, text_scale = 1, 
                                          color = line_color, text_offset = 10)
    box_annotator = sv.BoxAnnotator(thickness = 2, text_thickness = 1, text_scale = 0.5)
    # ---------------- zone definition ---------------- #

    model = YOLO(model_path, task = 'detect')
    # model.to(device)
    names = {0: 'btl', 1: 'hand'}
    in_count  = line_counter.in_count
    out_count = line_counter.out_count

    frame_length = 0
    id_tracking  = -1

    for result in model.track(url, imgsz = 128, conf = 0.4, device = device,
                              half = True, stream = True, persist = True, 
                              show = False, verbose = False, tracker = "./bytetrack_stream.yaml"):
                
        if frame_length == 0:
            start_time = time.time()
        # elif frame_length == 1800:
        #     break
        frame_length = frame_length + 1
        
        if result.boxes.id is None:
            pass
        else:
            frame = result.orig_img

            detections = sv.Detections.from_yolov8(result)
            detections = detections[detections.class_id == 0]
            detections.tracker_id = result.boxes.id.numpy().astype(int)
            # print(id_tracking, detections.tracker_id)
            line_counter.trigger(detections = detections)

            labels = [f"#{tracker_id} {names[class_id]} {confidence:0.2f}"
                                for _, _, confidence, class_id, tracker_id in detections]
            
            frame = box_annotator.annotate(scene = frame, detections = detections, labels = labels)
            line_annotator.annotate(frame = frame, line_counter = line_counter)
            
            if id_tracking != detections.tracker_id[0]:
                if id_tracking != -1:
                    out.release()
                    logging.info('video is saved as %s', filename)
                filename = save_dir + '/btl_' + str(detections.tracker_id[0]) + '.mp4'
                filename = get_unique_filename(filename)
                out = cv2.VideoWriter(filename, fourcc = fourcc,
                                      fps = 30, frameSize = (frame_width, frame_height))
            out.write(frame)
            id_tracking = detections.tracker_id[0]

            if in_count != line_counter.in_count:
                in_count = line_counter.in_count
                logging.info("count numbers in  %d", line_counter.in_count)
            elif out_count != line_counter.out_count:
                out_count = line_counter.out_count
                logging.info("count numbers out %d", line_counter.out_count)
            
    end_time = time.time()
    total_time = end_time - start_time
    logging.info('time passed       %.2f', total_time)
    logging.info('time on one frame %.2f', total_time/frame_length)
    logging.info('processing fps    %.2f', frame_length/total_time)
    logging.info('total numbers in  %d', line_counter.in_count)
    logging.info('total numbers out %d', line_counter.out_count)

if __name__ == '__main__':
    if torch.cuda.is_available():
        model_path = './weights/bestv8img128all.engine'
        device = torch.device(0)
    else:
        model_path = './weights/bestv8img128all.pt'
        device = 'cpu'
    
    video_path = 'rtsp://localhost:8554/entrance'
    dir = './demo'
    dir_list = [int(d[3:]) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d[0:3]=='run']
    save_dir = dir + '/run' + str(max(dir_list) + 1)
    os.makedirs(save_dir)

    camera_status = True

    logging.basicConfig(level=logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    log_file_handler = logging.FileHandler(save_dir + "/log.txt")
    log_file_handler.setFormatter(log_format)
    logging.getLogger().addHandler(log_file_handler)

    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    main(model_path, video_path, device, camera_status, save_dir)