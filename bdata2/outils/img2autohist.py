import cv2
import os

# 彩色图像进行自适应直方图均衡化
def hisEqulColor(im, clip = 2, grid = 8):
    ## 将RGB图像转换到YCrCb空间中
    img = im.copy()
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv2.split(ycrcb)
    # 以下代码详细注释见官网：
    # https://docs.opencv.org/4.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    clahe.apply(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def main():
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "img")
    save_dir = os.path.join(current_dir, "img_autohist")

    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") and filename[0:4] == '0206':
            img_path = os.path.join(data_dir, filename)
            # print(img_path)

            img = cv2.imread(img_path)
            img2 = hisEqulColor(img, clip = 5, grid = 10)
            save_path = os.path.join(save_dir, filename)
            # print(save_path)
            cv2.imwrite(save_path, img2)

if __name__ == '__main__':
    main()