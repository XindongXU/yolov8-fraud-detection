import cv2
import os

data_dir = '../demos_triche/00007_btl/run3'
imgs_dir = '../demos_triche/btl_19_img/'

data_dir = './demo/run7'
imgs_dir = './demo/btl_5_img/'

for filename in os.listdir(data_dir):
    if filename.endswith("20230911184435785_btl_5.mp4"):
        video_path = os.path.join(data_dir, filename)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        filename = filename[0:-4]
        print(filename)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(imgs_dir, f"{filename}_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        
        cap.release()
        print(f"save {frame_count} images to {filename}")