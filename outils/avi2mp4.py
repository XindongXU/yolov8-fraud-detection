import cv2

# 读取avi视频
avi_video = cv2.VideoCapture('0306_annot_30.avi')

# 获取视频的一些属性
fps = avi_video.get(cv2.CAP_PROP_FPS)
width = int(avi_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(avi_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建一个mp4格式的VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
mp4_video = cv2.VideoWriter('demo0306.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = avi_video.read()

    # 如果帧没有读取成功，跳出循环
    if not ret:
        break

    # 写入帧到mp4视频
    mp4_video.write(frame)

# 释放视频对象
avi_video.release()
mp4_video.release()
cv2.destroyAllWindows()
