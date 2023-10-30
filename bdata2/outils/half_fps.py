import cv2

video_path = "./video/202306161506_bbot00007.mp4"
output_path = "./video/202306161506_bbot00007_3fps.mp4"

video = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_path, fourcc, 3, (int(video.get(3)), int(video.get(4))))

frame_count = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    if frame_count % 3 == 0:
        output_video.write(frame)

    frame_count += 1

video.release()
output_video.release()
