import os
import shutil
import cv2


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


def copy_mp4_files(src_folder, dest_folder):

    # Créez le dossier de destination s'il n'existe pas
    os.makedirs(dest_folder, exist_ok=True)

    origin_list = os.listdir(db_folder[0:-6] + 'org')
    
    for filename in os.listdir(src_folder):
        if filename.endswith('.mp4'):
            # Construisez les chemins complets du fichier source et du fichier de destination
            id = origin_list.index(filename[0:18] + 'org' + filename[21:])
            src_file_path_1  = os.path.join(src_folder[0:-6] + 'org', origin_list[id-1])
            print(src_file_path_1)
            src_file_path_2  = os.path.join(src_folder[0:-6] + 'org', origin_list[id])
            dest_file_path   = os.path.join(dest_folder, filename[0:18] + 'org' + filename[21:])

            concatenate_videos(src_file_path_1, src_file_path_2, dest_file_path)
# Spécifiez les dossiers source et destination
db_folder = './00323-M/run2/triche'
dt_folder = './00323-M/run2/db'

copy_mp4_files(db_folder, dt_folder)