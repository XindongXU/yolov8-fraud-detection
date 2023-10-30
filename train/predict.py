from ultralytics import YOLO

# # Load a model
model = YOLO('./runs/detect/train_202306271822_n_s_dataset_6/weights/best.pt')  
# build from YAML and transfer weights

# model.predict(source = './datasets/dataset_0/train/0106_45.jpg', save=True, imgsz=320, conf=0.5)

# cap = cv2.VideoCapture("./video/test.mp4")
# cap.set (cv2.CAP_PROP_POS_FRAMES, 150)
# ret, frame = cap.read()
model.predict(source = './datasets/dataset_1/valid/0106_150.jpg', save=True, imgsz=320, conf=0.5)