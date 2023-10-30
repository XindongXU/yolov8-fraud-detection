import os
import shutil
import numpy as np
import argparse

def split_dataset(args):
    print(args)

    # specify the source data folder
    dataset_name = 'dataset_20230725'
    if args.image_hist:
        img_dir = './datasets_origin/' + dataset_name + '/img_autohist'
        ano_dir = './datasets_origin/' + dataset_name + '/anno'
    else:
        img_dir = './datasets_origin/' + dataset_name + '/img'
        ano_dir = './datasets_origin/' + dataset_name + '/anno'

    # specify training set and validation set folders
    datasets_dir = './datasets'
    dataset_list = [int(d[8:]) for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    dataset_num = max(dataset_list) + 1
    train_dir = './datasets/dataset_' + str(dataset_num) + '/train'
    valid_dir = './datasets/dataset_' + str(dataset_num) + '/valid'
    test_dir  = './datasets/dataset_' + str(dataset_num) + '/test'

    # create training set and validation set folders
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    # get the filenames of all images in the source folder (excluding extensions)
    img_list = [f[:-4] for f in os.listdir(img_dir) if f.endswith('.jpg')]
    print('length of original dataset', len(img_list))

    if args.shuffle_flag:
        print('shuffled')
        np.random.shuffle(img_list)

    # size of train dataset
    train_size = int(len(img_list) * args.split_ratio)
    print('length of train dataset', train_size)

    # assigning images and labels to the training and validation sets
    for i, img in enumerate(img_list):
        if i%100 == 0:
            print('split in processing :', i)
        if i < train_size:
            shutil.copy(os.path.join(img_dir, img+'.jpg'), os.path.join(train_dir, img+'.jpg'))
            if not os.path.isfile(os.path.join(ano_dir, img+'.txt')):
                # print(f"annotation file {os.path.join(ano_dir, img+'.txt')} doesnt exist for this image")
                continue
            shutil.copy(os.path.join(ano_dir, img+'.txt'), os.path.join(train_dir, img+'.txt'))
        else:
            shutil.copy(os.path.join(img_dir, img+'.jpg'), os.path.join(valid_dir, img+'.jpg'))
            if not os.path.isfile(os.path.join(ano_dir, img+'.txt')):
                continue
            shutil.copy(os.path.join(ano_dir, img+'.txt'), os.path.join(valid_dir, img+'.txt'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle_flag', action='store_false')
    parser.add_argument('--split_ratio',  default=0.9, type=float)
    parser.add_argument('--image_hist', action='store_false')


    args = parser.parse_args()

    split_dataset(args)

if __name__ == "__main__":
    main()
    # example of exécution
    # python split_dataset.py --split_ratio 0.9 --image_hist
    # shuffle + img + --split_ratio 0.9