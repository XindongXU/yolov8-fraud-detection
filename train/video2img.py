import cv2
import os

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "video")
imgs_dir = os.path.join(current_dir, "img")
# image_dir = os.path.join(current_dir, "data")
# os.makedirs(image_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.endswith(".mp4"):
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
        print(f"已保存 {frame_count} 帧图像从视频 {filename}")