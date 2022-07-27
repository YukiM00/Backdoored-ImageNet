import os,argparse
import pandas as pd
import numpy as np
import cv2
import scipy.io
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

FILE_NAME = 0
LABEL = 1

label2nb_dict = {
    'ChestX':{'NORMAL': 0, 'PNEUMONIA': 1}
    }

def make_chestx_data(df_files,data_dir,img_size=299):
    mapping = label2nb_dict['ChestX']
    X, y = [], []
    for idx, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
        img_path = os.path.join(data_dir, row[FILE_NAME])
        img = img_to_array(load_img(img_path, grayscale=True,
                                    color_mode='gray', target_size=(img_size,img_size)))
        X.append(img)
        y.append(mapping[row[LABEL]])
    X = np.asarray(X, dtype='float32')
    y = np.asarray(y)
    y = y.reshape(len(y), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    y = onehot_encoder.fit_transform(y)
    y = np.asarray(y, dtype='float32')
    return X, y

def humansize(nbytes):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

def make_imagenet_data(data_path='ImageNet_Validation/image_net',img_size=299):
    base_path = data_path +'/val/images'

    fns = os.listdir(base_path)
    fns.sort()
    fns = [
        base_path + '/' + fn
        for fn in fns
    ]

    x_val = np.zeros((len(fns), img_size, img_size, 3), dtype=np.float32)
    print(humansize(x_val.nbytes))

    for i in range(len(fns)):
        if i %2000 == 0:
            print("%d/%d" % (i, len(fns)))
        
        # Load (as BGR)
        img = cv2.imread(fns[i])
        
        # Resize
        height, width, _ = img.shape
        new_height = height * img_size // min(img.shape[:2])
        new_width = width * img_size // min(img.shape[:2])
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Crop
        height, width, _ = img.shape
        startx = width//2 - (img_size//2)
        starty = height//2 - (img_size//2)
        img = img[starty:starty+img_size,startx:startx+img_size]
        assert img.shape[0] == img_size and img.shape[1] == img_size, (img.shape, height, width)
        
        # Save (as RGB)
        x_val[i,:,:,:] = img[:,:,::-1]

    meta = scipy.io.loadmat(data_path+"/data/meta.mat")
    original_idx_to_synset = {}
    synset_to_name = {}

    for i in range(1000):
        ilsvrc2012_id = int(meta["synsets"][i,0][0][0][0])
        synset = meta["synsets"][i,0][1][0]
        name = meta["synsets"][i,0][2][0]
        original_idx_to_synset[ilsvrc2012_id] = synset
        synset_to_name[synset] = name

    synset_to_keras_idx = {}
    keras_idx_to_name = {}

    f = open(data_path+"/data/synset_words.txt","r")
    idx = 0
    for line in f:
        parts = line.split(" ")
        synset_to_keras_idx[parts[0]] = idx
        keras_idx_to_name[idx] = " ".join(parts[1:])
        idx += 1
    f.close()

    f = open(data_path+"/data/ILSVRC2012_validation_ground_truth.txt","r")
    y_val = f.read().strip().split("\n")
    y_val = list(map(int, y_val))
    y_val = np.array([synset_to_keras_idx[original_idx_to_synset[idx]] for idx in y_val])
    f.close()

    return x_val, y_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ChestX', type=str,help='ChestX ')
    parser.add_argument('--data_dir', default='User:CellData/chest_xray',type=str)
    parser.add_argument('--img_size', default=299, type=int, help='image size')
    args = parser.parse_args()
    print(args)

    if args.dataset == 'ChestX':
        df_train = pd.read_csv(
            './data/subdata/chestx_x_train_pre.csv',
            header=None)
        df_test = pd.read_csv(
            './data/subdata/chestx_x_test_pre.csv',
            header=None)
        X_train, y_train = make_chestx_data(
            df_files=df_train,
            img_dir=args.data_dir,
            img_size=args.img_size)
        X_test, y_test = make_chestx_data(
            df_files=df_test,
            img_dir=args.data_dir,
            img_size=args.img_size)
        np.save('./data/ChestX/X_train_{0}.npy'.format(args.img_size),X_train)
        np.save('./data/ChestX/X_test_{0}.npy'.format(args.img_size),X_test)
        np.save('./data/ChestX/y_train_{0}.npy'.format(args.img_size),y_train)
        np.save('./data/ChestX/y_test_{0}.npy'.format(args.img_size),y_test)
    elif args.dataset == 'ImageNet':
        X_val, y_val = make_imagenet_data(
            datadir=args.data_dir,
            img_size=args.data_dir)
        np.save('./data/ImageNet/val/X_val_{0}.npy'.format(args.img_size),X_val)
        np.save('./data/ImageNet/val/y_val_{0}.npy'.format(args.img_size),y_val)
    else:
        print("UNKNOWN DATASET")
            
