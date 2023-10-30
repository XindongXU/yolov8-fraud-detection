import cv2
import os


video_list = os.listdir('./00323-M/run2/db')

video_path = './00323-M/run2/db/' + '20231009204511554_org_22057.mp4'
cap1 = cv2.VideoCapture(video_path)
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap1.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./00323-M/run2/db/vrai.mp4', fourcc, fps, (width, height))

for video in video_list:
    video_path = './00323-M/run2/db/' + video
    print(video_path)
    cap1 = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        out.write(frame)

    cap1.release()

out.release()