import time
import cv2
import torch
from ultralytics import YOLO
import supervision as sv
print('sv.__version__', sv.__version__)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main(model_path, video_path, device, camera_status = True):
    # Infos concernantes cette exécution
    print('model used ', model_path)  # modèle utilisé
    print('video input', video_path)  # entrée vidéo
    print('device info', device)  # info appareil
    print('camera stat', camera_status)  # statut de la caméra

    # ---------------- zone definition ---------------- #
    # position de départ et de fin (1280, 720)
    if camera_status == 'xxl':
        LINE_START = sv.Point(700, 0)
        LINE_END = sv.Point(700, 1080)
    else:
        if camera_status:
            LINE_START = sv.Point(480, 0)
            LINE_END = sv.Point(480, 720)
        else:
            # sur la 00034, le sens de vue de caméra est inversé
            LINE_START = sv.Point(420, 480)
            LINE_END = sv.Point(420, 0)

    # compteur lié à la ligne de séparation
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    # ---------------- zone definition ---------------- #

    # ---------------- loading model ---------------- #
    model = YOLO(model_path, task = 'detect')
    # model.to(device)
    # ---------------- loading model ---------------- #

    # ---------------- count number ---------------- #
    in_count  = line_counter.in_count
    out_count = line_counter.out_count
    for result in model.track(video_path, imgsz = 128, conf = 0.4, device = device,
                              half = True, stream = True, persist = True, 
                              show = False, verbose = False, tracker = "./bytetrack_stream.yaml"):
        if result.boxes.id is None:
            # continuer si aucune détection
            continue
        else:
            detections = sv.Detections.from_yolov8(result) # détections à partir de yolov8
            # filtrer les détections, car on suit que les btls
            detections = detections[detections.class_id == 0]
            # id de suivi
            detections.tracker_id = result.boxes.id.numpy().astype(int)
            # déclencher le compteur
            line_counter.trigger(detections=detections)

            # Comparer le compteur d'entrées actuel avec le précédent
            if in_count != line_counter.in_count:
                in_count = line_counter.in_count
                print('count numbers in ', line_counter.in_count)
            # Comparer le compteur de sorties actuel avec le précédent
            elif out_count != line_counter.out_count:
                out_count = line_counter.out_count
                print('count numbers out', line_counter.out_count)
            
    print('count numbers in ', line_counter.in_count)
    print('count numbers out', line_counter.out_count)
    # ---------------- count number ---------------- #

if __name__ == '__main__':
    # Inférence du modèle sur jetson
    model_path = './weights/bestv8img128all.engine'
    video_path = 'rtsp://localhost:8554/entrance'
    device = torch.device(0)
    
    # camera_status doit être toujours True
    # car on a fixé la=e snes de caméra sur bbot34
    camera_status = True

    main(model_path, video_path, device, camera_status)