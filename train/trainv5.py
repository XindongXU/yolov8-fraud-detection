import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from ultralytics import YOLO
import torch
import yaml
import argparse
# torch.cuda.set_device(0)

def train(args):
    # write in data yaml file
    with open('./btl.yaml', 'r') as file:
        btl = yaml.safe_load(file)

    # assigning the training dataset, and compile data yaml file for training
    datasets_dir = './datasets'
    dataset_list = [int(d[8:]) for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    dataset_num = max(dataset_list)
    dataset_num = 6

    btl['train'] = './dataset_' + str(dataset_num) + '/train'
    btl['val'] = './dataset_' + str(dataset_num) + '/valid'
    btl['test'] = './dataset_' + str(dataset_num) + '/test'

    with open('./btl.yaml', 'w') as file:
        print('dataset yaml printing :')
        print(btl)
        yaml.safe_dump(btl, file)

    # Load a model
    # model_path = 'yolov8' + args.version + '.yaml'
    check_path = 'yolov5' + args.version + '.pt'
    print('loading weights from :', check_path)
    # model = YOLO(model_path)  # build a new model from YAML
    model = YOLO(check_path)  # load a pretrained model (recommended for training)
    # model = YOLO(model_path).load(check_path)  # build from YAML and transfer weights

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # select device
    # model.to(device)  # move model to selected device

    # Train the model
    model.train(data='./btl.yaml', epochs=500, imgsz=640, batch = -1, device=0)
    # input of the training images : 1280*720

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='n', type=str)
    # version choice of yolov8 can be in [n, s, m, l, x]
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()