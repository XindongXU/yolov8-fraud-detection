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

from mqtt_jetson import connect_mqtt, publish, subscribe, get_code_recu_flag

names = {0: 'btl', 1: 'main'}

version = '0.0.3'
'''
    2023-10-27 12:30
    MQTT ok :
    ping-pong + version
    bouteille ok (btl count in)
    bouteille retirée (btl count out)
    bouteille doublée
    bouteille vue
    bouteille disparue (btl disparue après scan)

    Remove camera_status flag
'''

def get_unique_filename(base_filename):
    """
    Retourne un nom de fichier basé sur base_filename qui n'existe pas encore.
    Si base_filename existe, il ajoute (1), (2), etc. 
    jusqu'à ce qu'un nom de fichier unique soit trouvé.
    """
    # Séparation du nom de fichier et de l'extension
    filename, extension = os.path.splitext(base_filename)

    counter = 1
    while os.path.exists(base_filename):
        base_filename = f"{filename}({counter}){extension}"
        counter += 1

    return base_filename

def concatenate_videos(video_path1, video_path2, output_path):

    # Ouverture du premier fichier vidéo
    cap1 = cv2.VideoCapture(video_path1)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    
    # Initialisation de l'écriture de la vidéo de sortie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # L'écriture de la première vidéo
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        out.write(frame)
    
    cap1.release()

    # Ouverture du deuxième fichier vidéo
    cap2 = cv2.VideoCapture(video_path2)

    # Lecture et écriture du deuxième fichier vidéo
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
        self.img_type = img_type  # 'img_ini, img_fin, img_in, img_out
        self.timestamp = timestamp  # Timestamp de l'image
        self.pos = pos  # Position de l'objet dans l'image
    
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

# onction principale
def main(model_path, video_path, device, save_dir, show_flag, mqtt_flag):

    # Accès à la variable globale names
    global names, version

    # -------------------------------- zone definition -------------------------------- #
    # Configuration de la connexion MQTT si le flag mqtt est activé
    if mqtt_flag:
        # Adresse du courtier MQTT (adresse IP de connection vers la bbot)
        broker = '192.168.1.163'
        port = 1883 # Port défaut du courtier MQTT
        # Sujets pour la publication et l'abonnement
        topic_pub = "bbot/ihm"
        topic_sub = "bbot/ia"
        topic_pin = "bbot/ping"
        topic_pon = "bbot/pong"

        client_id = f'Jetson'
        # initialise mqtt client connection
        client = connect_mqtt(broker, port, client_id)
        subscribe(client, topic_sub, topic_pin, topic_pon, version)
        client.loop_start() # Démarrer la boucle MQTT

    # Journalisation des informations sur le modèle, la vidéo et le dispositif
    logging.info("yolov8 info - %s", model_path)
    logging.info("video input - %s", video_path)
    logging.info("device info - %s", device)
    logging.info("demos saved - %s", save_dir)
    
    # Configuration de l'entrée du modèle (flux vidéo de ffmpeg sur jetson)
    url = video_path
    cap = cv2.VideoCapture(url)

    # Si la caméra est perdue, reboot le jetson automatiquement
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {url}, rebooting")
    
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))
    logging.info("stream  fps - %d", frame_fps)
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_fps = 30 if frame_fps == 90000 else frame_fps

    # Définition de la ligne séparation, taille de l'image originale (640, 480)
    LINE_START = sv.Point(240, 0)
    LINE_END = sv.Point(240, 480)

    # Définition du compteur et annotator liés à la ligne
    line_counter = sv.LineZone(start = LINE_START, end = LINE_END)
    line_color = Color(r = 224, g = 57, b = 151)
    line_annotator = sv.LineZoneAnnotator(thickness = 2, text_thickness = 2, text_scale = 1, 
                                          color = line_color, text_offset = 10)
    box_annotator = sv.BoxAnnotator(thickness = 2, text_thickness = 1, text_scale = 0.5)

    # Construction du modèle YOLO, en important les paramètres
    model = YOLO(model_path, task = 'detect')

    # Initialisation des variables modifiables dans la boucle d'analyse
    in_count  = line_counter.in_count
    out_count = line_counter.out_count

    frame_length = 0
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
    img_max_pos = 0
    filename_previous = None
    # -------------------------------- zone definition -------------------------------- #

    # Boucle d'analyse pour traiter chaque résultat de détection sur l'image
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
                if is_opened and is_ending > 0:
                    is_ending -= 1
                    frame_apres.append(frame0)

                # split video once no btl for {fix_length} consecutive images
                if is_opened and is_ending == 0:
                    
                    # write {fix_length = 10} images into 'out' and 'out_org' video
                    for frame_a in frame_apres:
                        out.write(frame_a)
                        out_org.write(frame_a)

                    is_ending = fix_length
                    frame_apres = []

                    # video cut by image vide
                    # logging.info('video is saved as %s', filename)
                    logging.info('video cut by img vide')
                    out.release()
                    out_org.release()

                    # check if this video contains double bouteille triche
                    if dbl_btl_flag == True:
                        
                        if mqtt_flag:
                            publish(client, topic_pub, msg = f"dbl btl-{filename_previous}-{datetime.now()}")
                        
                        logging.info("TRICHE POTENTIELLE 1: %s", filename_previous)
                        logging.info("TRICHE POTENTIELLE 2: %s", filename)
                        
                        video_path1 = filename_previous + '.mp4'
                        video_path2 = filename + '.mp4'
                        output_path = filename[0:len(save_dir)+1] + 'triche/' + filename[len(save_dir)+1:] + '.mp4'
                        concatenate_videos(video_path1, video_path2, output_path)

                    # check if this video contains bouteille disparue triche
                    # print('xy', np.min(detections[detections.class_id == 0].xyxy[:, [0, 2]]))
                    # print('code', get_code_recu_flag())
                    # print('img in', img_in_exit)
                    if get_code_recu_flag() and np.min(detections[detections.class_id == 0].xyxy[:, [0, 2]])<= 240 and img_in_exit == False:
                        get_code_recu_flag(change = True)
                        if mqtt_flag:
                            publish(client, topic_pub, msg = f"btl disparue-{datetime.now()}")
                        logging.info("1ERE BTL DISPARUE   : %s", filename)

                    info_video.tmp_fin = datetime.now()
                    # info_video.printout()

                    filename_previous = filename
                    filename = None
                    tab = {}
                pass
        
        # Objet(s) de cls==0 détectés par le modèle
        else:
            
            # Copie de frame originale, pour l'ajout des annotations sur l'image
            frame = frame0.copy()

            detections = sv.Detections.from_yolov8(result)
            detections.tracker_id = result.boxes.id.numpy().astype(int)

            # Ne suivre que les objets de (detections.class_id == 0)
            line_counter.trigger(detections = detections[detections.class_id == 0])

            # Ajout des annotations sur l'image
            labels = [f"#{tracker_id} {names[class_id]} {confidence:0.2f}"
                        for _, _, confidence, class_id, tracker_id in detections]
            frame = box_annotator.annotate(scene = frame, detections = detections, labels = labels)
            line_annotator.annotate(frame = frame, line_counter = line_counter)
            
            # check if reach the fin de video
            # separte video if new btl id is found in detections.tracker_id
            if len(tab) != 0 and set(tab.keys()).isdisjoint(set(detections.tracker_id)):

                # write {fix_length = 10} images into 'out' and 'out_org' video
                for frame_a in frame_apres:
                    out.write(frame_a)
                    out_org.write(frame_a)

                is_ending = fix_length
                frame_apres = []

                # video cut by new btl introduite
                # logging.info('video is saved as %s', filename)
                logging.info('video cut by new btl id found')
                out.release()
                out_org.release()

                # check if this video contains double bouteille triche
                if dbl_btl_flag == True:
                    
                    if mqtt_flag:
                        publish(client, topic_pub, msg = f"dbl btl-{filename_previous}-{datetime.now()}")

                    logging.info("TRICHE POTENTIELLE 1: %s", filename_previous)
                    logging.info("TRICHE POTENTIELLE 2: %s", filename)

                    video_path1 = filename_previous + '.mp4'
                    video_path2 = filename + '.mp4'
                    output_path = filename[0:len(save_dir)+1] + 'triche/' + filename[len(save_dir)+1:] + '.mp4'
                    concatenate_videos(video_path1, video_path2, output_path)

                info_video.tmp_fin = datetime.now()
                # info_video.printout()

                filename_previous = filename
                filename = None
                tab = {}

            # début de video, car tab n'existe jamais ou tab est initalisé à vide set
            if len(tab) == 0 and len(detections[detections.class_id == 0].xyxy) > 0:
                current_time = datetime.now()
                # Création de fichier vidéo
                filename = (save_dir + '/'
                            + str(current_time.strftime('%Y%m%d%H%M%S')) + str(current_time.microsecond)[:3]
                            + '_btl_' + str(detections.tracker_id[0]))
                
                if mqtt_flag:
                    publish(client, topic_pub, msg = f"btl vue-{filename}-{current_time}")

                out = cv2.VideoWriter(filename + '.mp4', fourcc = fourcc,
                                      fps = frame_fps, frameSize = (frame_width, frame_height))
                
                filename_org = (save_dir + '/org/'
                            + str(current_time.strftime('%Y%m%d%H%M%S')) + str(current_time.microsecond)[:3]
                            + '_org_' + str(detections.tracker_id[0]))
                out_org = cv2.VideoWriter(filename_org + '.mp4', fourcc = fourcc,
                                      fps = frame_fps, frameSize = (frame_width, frame_height))
                
                # write {fix_length = 10} images into 'out' and 'out_org' video
                for frame_a in frame_avant:
                    out.write(frame_a)
                    out_org.write(frame_a)
                
                # Création des infos à noter concernant la vidéo et les objets détectés
                info_video = InfoVideo(tab)
                info_video.obj_ids.update(detections.tracker_id)
                for index, id in enumerate(detections.tracker_id):
                    info_video.obj.append(Objet(detections.class_id[index], id, detections.xyxy[index]))
                    tab[id] = len(info_video.obj) - 1

                # Vérifie si la vidéo contient une double bouteille triche
                fst_btl_triche = False
                dbl_btl_flag = False
                # print(abs(img_ini_tmp - datetime.now()).total_seconds())
                # print(img_in_exit)
                # print(img_ini_pos)
                if min(img_ini_pos[0], img_ini_pos[2]) <= 240 and img_max_pos >= 100 and abs(img_ini_tmp - datetime.now()).total_seconds() <= 3 and img_in_exit == False:
                    fst_btl_triche = True
                else:
                    fst_btl_triche = False

                img_ini_tmp = datetime.now()
                img_ini_pos = detections[detections.class_id == 0].xyxy[0]
                fst_btl_id  = detections[detections.class_id == 0].tracker_id[0]
                if len(detections[detections.class_id == 0].tracker_id) > 1:
                    for index, id in enumerate(detections[detections.class_id == 0].tracker_id):
                        if min(detections[detections.class_id == 0].xyxy[index][0], detections[detections.class_id == 0].xyxy[index][2]) <= min(img_ini_pos[0], img_ini_pos[2]):
                            img_ini_pos = detections[detections.class_id == 0].xyxy[index]
                            fst_btl_id  = id
                img_max_pos = max(img_ini_pos[0], img_ini_pos[2])
                # print(img_ini_pos)
                img_in_exit = False

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
                        if current_id == fst_btl_id:
                            pos = detections.xyxy[index]
                            img_max_pos = max(img_max_pos, max(pos[0], pos[2]))
            
            out.write(frame)
            out_org.write(frame0)

            # Comparer le compteur d'entrées actuel avec le précédent
            if in_count != line_counter.in_count:
                img_in_exit = True

                # Mettre à jour le compte d'entrée
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

            # Comparer le compteur de sorties actuel avec le précédent
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

            # logging.info('video is saved as %s', filename)
            logging.info('video cut by end of for loop')
            
            out.release()
            out_org.release()
            
            # check if this video contains double bouteille triche
            if dbl_btl_flag == True:
                
                if mqtt_flag:
                    publish(client, topic_pub, msg = f"dbl btl-{filename_previous}-{datetime.now()}")
                
                logging.info("TRICHE POTENTIELLE 1: %s", filename_previous)
                logging.info("TRICHE POTENTIELLE 2: %s", filename)
                
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
    if mqtt_flag:
        client.loop_stop()
    logging.info('time passed       %.2f', total_time)
    logging.info('time on one frame %.2f', total_time/frame_length)
    logging.info('processing fps    %.2f', frame_length/total_time)
    logging.info('total numbers in  %d', line_counter.in_count)
    logging.info('total numbers out %d', line_counter.out_count)

if __name__ == '__main__':
    
    # Si CUDA est disponible, on est sur jetson
    if torch.cuda.is_available():
        # Sur jetson le modèle est accéléré par TensorRT engine
        model_path = './weights/bestv8img128all.engine'
        device = torch.device(0)

        video_path = 'rtsp://localhost:8554/entrance'
        # video_path = './video/faux.mp4'
        # video_path = './video/vrai.mp4'

        show_flag = False
        mqtt_flag = True
    
    # Inférence du modèle sur l'ordi local
    else:
        model_path = './weights/bestv8img128all.pt'
        device = 'cpu'
        # Utiliser le CPU si CUDA n'est pas disponible

        show_flag = True
        mqtt_flag = False

        video_path = './00323-M/run2/db/202309151402_bbot00034_btl_dcp.mp4'
        video_path = './00323-M/run2/db/faux.mp4'
    
    # Enregistrement des résultats de toutes les exécutions du modèle dans ./demo
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

    # Pour chaque exécution, on crée un nouveau dossier 'run'
    save_dir = dir + '/run' + str(max_number + 1)
    # Dans 'run', on crée les dossiers /org, /img, /triche
    os.makedirs(save_dir)
    # Enregistrement des images au moment de btl passage
    os.makedirs(save_dir + '/img')
    # Enregistrement des vidéos de triche (double bouteilles)
    os.makedirs(save_dir + '/triche')
    # Enregistrement des vidéos pures de triche
    os.makedirs(save_dir + '/org')

    # Configuration de logging système
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

    main(model_path, video_path, device, save_dir, show_flag, mqtt_flag)