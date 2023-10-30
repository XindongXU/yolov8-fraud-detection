import cv2
from ultralytics import YOLO
from datetime import datetime
from img2autohist import hisEqulColor

show_flag = False
# test_file = "./video/202306161449_bbot00007.mp4"
# test_file = "./video/202306161557_bbot00036.mp4"
test_file = "./video/202306151526_bbot00005.mp4"

cap = cv2.VideoCapture(test_file)

model = YOLO('./runs/detect/train_202306271822_n_s_dataset_6/weights/best.pt')
# build from YAML and transfer weights

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frame_length', frame_length)

# create VideoWriter object, set output name, encoder format, fps and w+h
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S")
filename = filename + '_bbot000' + test_file[-6:-4] + '.avi'
out = cv2.VideoWriter('./demo/demo_' + filename, cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width, frame_height))

frame_id = 0
while True:
    print('frame_id', frame_id)
    ret, frame = cap.read()
    if not ret:
        break
    frame_id = frame_id + 1
    frame = hisEqulColor(frame, clip = 5, grid = 10)
    results = model.track(frame, imgsz=640, persist=True)
    annotated_frame = results[0].plot()
    # if results[0].boxes.id is not None:  # add this check
    #     boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    #     ids = results[0].boxes.id.cpu().numpy().astype(int)
    #     for box, id in zip(boxes, ids):
    #         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #         cv2.putText(
    #             frame,
    #             f"Id {id}",
    #             (box[0], box[1]),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             1,
    #             (0, 0, 255),
    #             2,
    #         )
    # write frames into the output video
    out.write(annotated_frame)
    if show_flag == True:
        cv2.imshow("annotated_frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()