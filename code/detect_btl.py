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
import numpy as np

names = {0: 'btl', 1: 'main'}

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

class InfoImg:
    def __init__(self, img_type, timestamp, pos) -> None:
        self.img_type = img_type
        # 'img_ini, img_fin, img_in, img_out
        self.timestamp = timestamp
        self.pos = pos
    
    def printout(self):
        logging.info("-------- image info presents as follow --------")
        logging.info("obj stats - %s", self.img_type)
        logging.info("image  at - %s", self.timestamp)
        logging.info("obj is at - %s", self.pos)

class Objet:
    def __init__(self, class_id, id, pos_ini):
        global names
        self.class_id = names[class_id]
        self.id = id
        self.img_ini = InfoImg(img_type = "img_ini", 
                               timestamp = datetime.now(), 
                               pos = pos_ini)
        self.img_fin = InfoImg(img_type = "img_fin",
                               timestamp = datetime.now(), 
                               pos = pos_ini)
        self.img_in_out = []

class InfoVideo:
    def __init__(self, tab):
        self.tab = tab
        self.tmp_ini = datetime.now()
        self.tmp_fin = datetime.now()
        self.obj_ids = set()
        self.obj = []

    @property
    def obj_nbs(self):
        return len(self.obj_ids)
    
    def printout(self):
        logging.info("---------------- video info presents as follow ----------------")
        logging.info("start  at - %s", self.tmp_ini)
        logging.info("finish at - %s", self.tmp_fin)
        logging.info("id of obj - %s", self.obj_ids)
        logging.info("nb of obj - %d", self.obj_nbs)
        # print(self.tab)
        for i, item in enumerate(self.obj_ids):
            logging.info("class obj - %s", self.obj[self.tab[item]].class_id)
            logging.info("id of obj - %d", self.obj[self.tab[item]].id)
            self.obj[self.tab[item]].img_ini.printout()
            self.obj[self.tab[item]].img_fin.printout()
            if len(self.obj[self.tab[item]].img_in_out):
                for j, img in enumerate(self.obj[self.tab[item]].img_in_out):
                    img.printout()
            else:
                logging.info("obj stats - Cette %s n'a jamais franchi", self.obj[self.tab[item]].class_id)
                
        logging.info("---------------------------------------------------------------")

def main(model_path, video_path, device, camera_status, save_dir):

    global names
    
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
    logging.info("stream  fps - %d", frame_fps)
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_fps = 30 if frame_fps == 90000 else frame_fps

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
    
    in_count  = line_counter.in_count
    out_count = line_counter.out_count

    frame_length = 0
    id_tracking  = -1
    tab = {}
    frame_avant = []
    frame_apres = []
    fix_length = 10
    is_ending  = fix_length

    for result in model.track(url, imgsz = 128, conf = 0.4, device = device,
                              half = True, stream = True, persist = True, 
                              show = False, verbose = False, tracker = "./bytetrack_stream.yaml"):
        
        if frame_length == 0:
            start_time = time.time()
        # elif frame_length == 600:
        #     break
        frame_length = frame_length + 1
        
        frame0 = result.orig_img

        # if is_ending < fix_length:
        #     frame_apres.append(frame_org)
        #     is_ending -= 1
        # if is_ending == -1:
        #     # print(len(frame_apres), frame_apres)
        #     out.write(np.dstack(frame_apres))
        #     frame_apres = []
        #     is_ending = fix_length
        #     out.release()
        #     out_org.release()

        #     filename = None
        #     id_tracking = -1
        #     tab = {}

        # check no object detected or no btl(cls==0) detected
        if result.boxes.id is None or result.boxes.id[(result.boxes.cls == 0).nonzero()] is None:
            
            try:
                is_opened = out.isOpened()
            except UnboundLocalError:
                # video 'out' not defined yet
                # video not being recorded
                pass
            # save 10(fix_length) images into frame_apres
            else:
                if is_opened and is_ending > 0:
                    is_ending -= 1
                    frame_apres.append(frame0)

                # split video once no btl for "fix_length" consecutive images
                if is_opened and is_ending == 0:

                    # print('len frame_apres', len(frame_apres))
                    for frame_a in frame_apres:
                        out.write(frame_a)
                    is_ending = fix_length
                    frame_apres = []

                    logging.info('video is saved as %s', filename)
                    info_video.tmp_fin = datetime.now()
                    info_video.printout()

                    # video cut by image vide
                    out.release()
                    out_org.release()
                    filename = None
                    id_tracking = -1
                    tab = {}
                pass
        # frame0 = result.orig_img

        else:
            # frame_org = result.orig_img
            frame = frame0.copy()

            detections = sv.Detections.from_yolov8(result)
            detections.tracker_id = result.boxes.id.numpy().astype(int)
            # only track btls
            line_counter.trigger(detections = detections[detections.class_id == 0])

            labels = [f"#{tracker_id} {names[class_id]} {confidence:0.2f}"
                                for _, _, confidence, class_id, tracker_id in detections]
            
            frame = box_annotator.annotate(scene = frame, detections = detections, labels = labels)
            line_annotator.annotate(frame = frame, line_counter = line_counter)
            
            # fin de video ?
            if id_tracking != -1 and id_tracking not in detections.tracker_id:
                # print('len frame_apres', len(frame_apres))
                # for frame_a in frame_apres[::-1]:
                for frame_a in frame_apres:
                    out.write(frame_a)
                is_ending = fix_length
                frame_apres = []

                logging.info('video is saved as %s', filename)
                info_video.tmp_fin = datetime.now()
                info_video.printout()

                # video cut by new btl introduite
                out.release()
                out_org.release()
                filename = None
                id_tracking = -1
                tab = {}

            # début de video
            if id_tracking == -1:
                current_time = datetime.now()
                filename = (save_dir + '/'
                            + str(current_time.strftime('%Y%m%d%H%M%S')) + str(current_time.microsecond)[:3]
                            + '_btl_' + str(detections.tracker_id[0]) + '.mp4')
                out = cv2.VideoWriter(filename, fourcc = fourcc,
                                      fps = frame_fps, frameSize = (frame_width, frame_height))
                # print('len frame_avant', len(frame_avant))
                for frame_a in frame_avant:
                    out.write(frame_a)
                
                filename_org = (save_dir + '/org/'
                            + str(current_time.strftime('%Y%m%d%H%M%S')) + str(current_time.microsecond)[:3]
                            + '_org_' + str(detections.tracker_id[0]) + '.mp4')
                out_org = cv2.VideoWriter(filename_org, fourcc = fourcc,
                                      fps = frame_fps, frameSize = (frame_width, frame_height))
                
                info_video = InfoVideo(tab)
                info_video.obj_ids.update(detections.tracker_id)
                for index, id in enumerate(detections.tracker_id):
                    info_video.obj.append(Objet(detections.class_id[index], id, detections.xyxy[index]))
                    tab[id] = len(info_video.obj) - 1

                id_tracking = detections.tracker_id[0]

            # enregistrement de video en cours / fichier "out" déjà existé
            else:
                is_ending = fix_length
                frame_apres = []
                if_existed = [id in info_video.obj_ids for id in detections.tracker_id]
                info_video.obj_ids.update(detections.tracker_id)

                for index, item in enumerate(if_existed):
                    # pour les ids jamais vus
                    if not item:
                        # info_video.obj_ids.update(detections.tracker_id)
                        info_video.obj.append(Objet(detections.class_id[index], detections.tracker_id[index], detections.xyxy[index]))
                        tab[detections.tracker_id[index]] = len(info_video.obj) - 1
                    # pour les ids déjà existés
                    else:
                        current_id = detections.tracker_id[index]
                        info_video.obj[tab[current_id]].img_fin = InfoImg(img_type = "img_fin",
                                                                          timestamp = datetime.now(),
                                                                          pos = detections.xyxy[index])
            
            out.write(frame)
            out_org.write(frame0)

            if in_count != line_counter.in_count:
                in_count = line_counter.in_count
                logging.info("count numbers in  %d", line_counter.in_count)
                info_video.obj[tab[detections.tracker_id[0]]].img_in_out.append(InfoImg("img_in",
                                                                                         datetime.now(),
                                                                                         detections.xyxy[0]))
            elif out_count != line_counter.out_count:
                out_count = line_counter.out_count
                logging.info("count numbers out %d", line_counter.out_count)
                info_video.obj[tab[detections.tracker_id[0]]].img_in_out.append(InfoImg("img_out",
                                                                                         datetime.now(),
                                                                                         detections.xyxy[0]))
        # enregistrement des images avant la video
        frame_avant.append(frame0)
        while len(frame_avant) > fix_length:
            frame_avant.pop(0)

    try:
        is_opened = out.isOpened()
    except UnboundLocalError:
        # print("Video 'out' is not defined yet")
        pass
    else:
        if is_opened:
            logging.info('video is saved as %s', filename)
            info_video.tmp_fin = datetime.now()
            info_video.printout()
            out.release()
            out_org.release()
            filename = None
            id_tracking = -1
            tab = {}
        else:
            pass 
    
    end_time = time.time()
    total_time = end_time - start_time
    logging.info('time passed       %.2f', total_time)
    logging.info('time on one frame %.2f', total_time/frame_length)
    logging.info('processing fps    %.2f', frame_length/total_time)
    logging.info('total numbers in  %d', line_counter.in_count)
    logging.info('total numbers out %d', line_counter.out_count)

if __name__ == '__main__':

    video_path = 'rtsp://localhost:8554/entrance'
    camera_status = True
    
    if torch.cuda.is_available():
        model_path = './weights/bestv8img128all.engine'
        device = torch.device(0)
        video_path = './video/20230818152928_bbot00034.mp4'
        camera_status = False
    else:
        model_path = './weights/bestv8img128all.pt'
        device = 'cpu'
        video_path = './video/20230818152928_bbot00034.mp4'
        camera_status = False
    
    dir = './demo'
    dir_list = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d.startswith('run')]

    max_number = -1
    if len(dir_list):
        for folder in dir_list:
            try:
                number = int(folder[3:])
                if number > max_number:
                    max_number = number
            except ValueError:
                pass

    save_dir = dir + '/run' + str(max_number + 1)
    os.makedirs(save_dir)
    os.makedirs(save_dir + '/org')

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