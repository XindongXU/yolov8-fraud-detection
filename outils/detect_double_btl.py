import time
import cv2
import torch
from ultralytics import YOLO
import supervision as sv
from supervision.draw.color import Color
from datetime import datetime
import os
import shutil
import sys
import logging
import numpy as np

from mqtt_jetson import connect_mqtt, publish, subscribe, subscribe_pingpong

names = {0: 'btl', 1: 'main'}

version = '0.0.1'
'''
    ping-pong + version check
    bouteille ok (btl count in)
    bouteille retirée (btl count out)
    double bouteille
    bouteille vue
    bouteille disparue (btl disparue après scan)
'''

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

def concatenate_videos(video_path1, video_path2, output_path):

    cap1 = cv2.VideoCapture(video_path1)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        out.write(frame)
    
    cap1.release()

    cap2 = cv2.VideoCapture(video_path2)

    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        out.write(frame)

    cap2.release()
    out.release()

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

def main(model_path, video_path, device, camera_status, save_dir, show_flag, mqtt_flag):

    global names
    
    if mqtt_flag:

        broker = '192.168.2.89'
        port = 1883
        topic_pub = "bbot/ihm"
        topic_sub = "bbot/ia"
        topic_pin = "bbot/ping"
        topic_pon = "bbot/pong"

        client_id = f'jetson-00034'

        client = connect_mqtt(broker, port, client_id)
        subscribe(client, topic_sub)
        subscribe_pingpong(client, topic_pin, topic_pon, version)
        client.loop_start()

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
    # id_tracking  = -1
    tab = {}
    frame_avant = []
    frame_apres = []
    fix_length = 10
    is_ending  = fix_length

    dbl_btl_flag = False
    fst_btl_triche = False
    img_in_exit = False
    img_ini_tmp = datetime(1999, 10, 11)
    img_ini_pos = [0, 0, 0, 0]
    filename_previous = None

    for result in model.track(url, imgsz = 128, conf = 0.4, device = device,
                              half = False, stream = True, persist = True, 
                              show = show_flag, verbose = False, tracker = "./bytetrack_stream.yaml"):
        
        # count frame number from the second image
        # to calculate treatment fps at the end of for lop
        if frame_length == 2:
            start_time = time.time()
        # elif frame_length == 600:
        #     break
        frame_length = frame_length + 1
        
        # save current pure img to frame0
        # prepare for img lists : 'frame_apres' and 'frame_avant'
        frame0 = result.orig_img

        # check no object detected or no btl(cls==0) detected
        # save these no objet frames {fix_length} as the begining or the end of one video
        # check whether to cut and save the current recording video, once no btl for {fix_length} consecutive images
        if result.boxes.id is None or result.boxes.id[(result.boxes.cls == 0).nonzero()] is None:
            
            # check if video 'out' is being recorded
            try:
                is_opened = out.isOpened()

            except UnboundLocalError:
                # video 'out' not defined yet
                # video not being recorded
                pass
            
            # video 'out' is being recorded
            # save {fix_length = 10} images into frame_apres
            # make {fix_length} minus 1, which makes {fix_length} rest images to be added into 'frame_après'
            else:
                if is_ending > 0:
                    is_ending -= 1
                    frame_apres.append(frame0)

                # split video once no btl for {fix_length} consecutive images
                if is_ending == 0:
                    
                    # write {fix_length = 10} images into 'out' and 'out_org' video
                    for frame_a in frame_apres:
                        out.write(frame_a)
                        out_org.write(frame_a)

                    is_ending = fix_length
                    frame_apres = []

                    # video cut by image vide
                    logging.info('video is saved as %s', filename)
                    out.release()
                    out_org.release()

                    # check if this video contains double bouteille triche
                    if dbl_btl_flag == True:
                        if mqtt_flag:
                            publish(client, topic_pub, msg = f"dbl btl-{filename_previous}-{datetime.now()}")
                        logging.info("TRICHE POTENTIELLE 1: %s", filename_previous)
                        logging.info("TRICHE POTENTIELLE 2: %s", filename)
                        # shutil.copy2(filename + '.mp4', filename[0:len(save_dir)+1] + 'triche/' + filename[len(save_dir)+1:] + '.mp4')
                        # shutil.copy2(filename_previous + '.mp4', filename_previous[0:len(save_dir)+1] + 'triche/' + filename_previous[len(save_dir)+1:] + '.mp4')
                        video_path1 = filename_previous + '.mp4'
                        video_path2 = filename + '.mp4'
                        output_path = filename[0:len(save_dir)+1] + 'triche/' + filename[len(save_dir)+1:] + '.mp4'
                        concatenate_videos(video_path1, video_path2, output_path)

                    info_video.tmp_fin = datetime.now()
                    # info_video.printout()

                    filename_previous = filename
                    filename = None
                    # id_tracking = -1
                    tab = {}
                pass

        else:
            
            frame = frame0.copy()

            detections = sv.Detections.from_yolov8(result)
            detections.tracker_id = result.boxes.id.numpy().astype(int)
            # only track btls (detections.class_id == 0)
            line_counter.trigger(detections = detections[detections.class_id == 0])

            labels = [f"#{tracker_id} {names[class_id]} {confidence:0.2f}"
                                for _, _, confidence, class_id, tracker_id in detections]
            frame = box_annotator.annotate(scene = frame, detections = detections, labels = labels)
            line_annotator.annotate(frame = frame, line_counter = line_counter)
            
            # check if reach the fin de video
            # separte video if new btl id is found in detections.tracker_id
            if len(tab) != 0 and set(tab.keys()).isdisjoint(set(detections.tracker_id)):
                # for frame_a in frame_apres[::-1]:
                for frame_a in frame_apres:
                    out.write(frame_a)
                    out_org.write(frame_a)

                is_ending = fix_length
                frame_apres = []

                # video cut by new btl introduite
                logging.info('video is saved as %s', filename)              
                out.release()
                out_org.release()

                # check if this video contains double bouteille triche
                if dbl_btl_flag == True:
                    if mqtt_flag:
                        publish(client, topic_pub, msg = f"dbl btl-{filename_previous}-{datetime.now()}")
                    logging.info("TRICHE POTENTIELLE 1: %s", filename_previous)
                    logging.info("TRICHE POTENTIELLE 2: %s", filename)
                    # shutil.copy2(filename + '.mp4', filename[0:len(save_dir)+1] + 'triche/' + filename[len(save_dir)+1:] + '.mp4')
                    # shutil.copy2(filename_previous + '.mp4', filename_previous[0:len(save_dir)+1] + 'triche/' + filename_previous[len(save_dir)+1:] + '.mp4')
                    video_path1 = filename_previous + '.mp4'
                    video_path2 = filename + '.mp4'
                    output_path = filename[0:len(save_dir)+1] + 'triche/' + filename[len(save_dir)+1:] + '.mp4'
                    concatenate_videos(video_path1, video_path2, output_path)

                info_video.tmp_fin = datetime.now()
                # info_video.printout()

                filename_previous = filename
                filename = None
                # id_tracking = -1
                tab = {}

            # début de video, because tab never exists or tab is initalised to vide set
            if len(tab) == 0:
                current_time = datetime.now()
                if mqtt_flag:
                    publish(client, topic_pub, msg = f"btl vue-{current_time}")
                filename = (save_dir + '/'
                            + str(current_time.strftime('%Y%m%d%H%M%S')) + str(current_time.microsecond)[:3]
                            + '_btl_' + str(detections.tracker_id[0]))
                out = cv2.VideoWriter(filename + '.mp4', fourcc = fourcc,
                                      fps = frame_fps, frameSize = (frame_width, frame_height))
                
                filename_org = (save_dir + '/org/'
                            + str(current_time.strftime('%Y%m%d%H%M%S')) + str(current_time.microsecond)[:3]
                            + '_org_' + str(detections.tracker_id[0]))
                out_org = cv2.VideoWriter(filename_org + '.mp4', fourcc = fourcc,
                                      fps = frame_fps, frameSize = (frame_width, frame_height))
                
                for frame_a in frame_avant:
                    out.write(frame_a)
                    out_org.write(frame_a)
                
                info_video = InfoVideo(tab)
                info_video.obj_ids.update(detections.tracker_id)
                for index, id in enumerate(detections.tracker_id):
                    info_video.obj.append(Objet(detections.class_id[index], id, detections.xyxy[index]))
                    tab[id] = len(info_video.obj) - 1

                fst_btl_triche = False
                dbl_btl_flag = False
                # print(abs(img_ini_tmp - datetime.now()).total_seconds())
                # print(img_in_exit)
                # print(img_ini_pos)
                if camera_status:
                    if min(img_ini_pos[0], img_ini_pos[2]) <= 240 and abs(img_ini_tmp - datetime.now()).total_seconds() <= 3 and img_in_exit == False:
                        fst_btl_triche = True
                    else:
                        fst_btl_triche = False
                else:
                    if max(img_ini_pos[0], img_ini_pos[2]) >= 420 and abs(img_ini_tmp - datetime.now()).total_seconds() <= 3 and img_in_exit == False:
                        fst_btl_triche = True
                    else:
                        fst_btl_triche = False
                img_ini_tmp = datetime.now()
                img_ini_pos = detections.xyxy[0]
                # print(img_ini_pos)
                img_in_exit = False
                # id_tracking = detections.tracker_id[0]

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
                img_in_exit = True

                in_count = line_counter.in_count
                if mqtt_flag:
                    publish(client, topic_pub, msg = f"btl ok-{datetime.now()}")
                logging.info("count numbers in  %d", line_counter.in_count)
                info_video.obj[tab[detections.tracker_id[0]]].img_in_out.append(InfoImg("img_in",
                                                                                         datetime.now(),
                                                                                         detections.xyxy[0]))
                current_time = datetime.now()
                file_img = (save_dir + '/img/'
                            + str(current_time.strftime('%Y%m%d%H%M%S')) + str(current_time.microsecond)[:3]
                            + '_btl_' + str(detections.tracker_id[0]))
                cv2.imwrite(file_img + '_in.png', frame)

                if fst_btl_triche == True:
                    fst_btl_triche = False
                    dbl_btl_flag = True
                
            elif out_count != line_counter.out_count:
                img_in_exit = True

                out_count = line_counter.out_count
                if mqtt_flag:
                    publish(client, topic_pub, msg = f"btl retirée-{datetime.now()}")
                logging.info("count numbers out %d", line_counter.out_count)
                info_video.obj[tab[detections.tracker_id[0]]].img_in_out.append(InfoImg("img_out",
                                                                                         datetime.now(),
                                                                                         detections.xyxy[0]))
                current_time = datetime.now()
                file_img = (save_dir + '/img/'
                            + str(current_time.strftime('%Y%m%d%H%M%S')) + str(current_time.microsecond)[:3]
                            + '_btl_' + str(detections.tracker_id[0]))
                cv2.imwrite(file_img + '_out.png', frame)

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
            for frame_a in frame_apres:
                out.write(frame_a)
                out_org.write(frame_a)

            is_ending = fix_length
            frame_apres = []

            logging.info('video is saved as %s', filename)
            
            out.release()
            out_org.release()
            
            if dbl_btl_flag == True:
                if mqtt_flag:
                    publish(client, topic_pub, msg = f"dbl btl-{filename_previous}-{datetime.now()}")
                logging.info("TRICHE POTENTIELLE 1: %s", filename_previous)
                logging.info("TRICHE POTENTIELLE 2: %s", filename)
                # shutil.copy2(filename + '.mp4', filename[0:len(save_dir)+1] + 'triche/' + filename[len(save_dir)+1:] + '.mp4')
                # shutil.copy2(filename_previous + '.mp4', filename_previous[0:len(save_dir)+1] + 'triche/' + filename_previous[len(save_dir)+1:] + '.mp4')
                video_path1 = filename_previous + '.mp4'
                video_path2 = filename + '.mp4'
                output_path = filename[0:len(save_dir)+1] + 'triche/' + filename[len(save_dir)+1:] + '.mp4'
                concatenate_videos(video_path1, video_path2, output_path)
            
            info_video.tmp_fin = datetime.now()
            # info_video.printout()
            
            filename_previous = filename
            filename = None
            # id_tracking = -1
            tab = {}
        else:
            pass 
    
    end_time = time.time()
    total_time = end_time - start_time
    client.loop_stop()
    logging.info('time passed       %.2f', total_time)
    logging.info('time on one frame %.2f', total_time/frame_length)
    logging.info('processing fps    %.2f', frame_length/total_time)
    logging.info('total numbers in  %d', line_counter.in_count)
    logging.info('total numbers out %d', line_counter.out_count)

if __name__ == '__main__':
    
    if torch.cuda.is_available():
        model_path = './weights/bestv8img128all.engine'
        device = torch.device(0)

        # video_path = './00323-M/run2/db/all.mp4'
        video_path = 'rtsp://localhost:8554/entrance'

        show_flag = False
        mqtt_flag = True
        camera_status = False
        
    else:
        model_path = './weights/bestv8img128all.pt'
        device = 'cpu'
        show_flag = True
        mqtt_flag = False

        video_path = './00323-M/run2/db/202309151402_bbot00034_btl_dcp.mp4'
        camera_status = False
        video_path = './00323-M/run2/db/vrai.mp4'
        camera_status = True
    
    dir = './demo'
    os.makedirs(dir, exist_ok=True)
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
    os.makedirs(save_dir + '/img')
    os.makedirs(save_dir + '/triche')

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

    main(model_path, video_path, device, camera_status, save_dir, show_flag, mqtt_flag)