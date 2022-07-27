import os
import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from utils.poison import make_trigger,make_trigger_label
import random
import pandas as pd
from tqdm import tqdm

from keras.applications.inception_v3 import preprocess_input as inceptionv3_input
from keras.applications.resnet50 import preprocess_input as resnet50_input
from keras.applications.densenet import preprocess_input as densenet_input
from keras.applications.xception import preprocess_input as xception_input
from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_v2_input
from keras.applications.vgg16 import preprocess_input as vgg16_input
from keras.applications.vgg19 import preprocess_input as vgg19_input
from keras.applications.mobilenet import preprocess_input as mobilenet_input

def get_images(subject_path):
    return sorted(os.listdir(subject_path))

def get_subjects(dataset_path):
    return sorted(os.listdir(dataset_path))

def load_utkface_data(dataset_path,csv_path,img_size):
    random.seed(43)
    image_data = []
    data = pd.read_csv(csv_path)

    for file in list(data['file']):
        image = cv2.resize(cv2.imread(dataset_path+file), (img_size, img_size))
        image_array = np.array(image,dtype=np.float32)
        image_data.append(image_array)
    
    image_data = np.array(image_data,dtype=np.float32)
    image_data = image_data[:,:,:,::-1]
    labels = list(data['label'])
    labels = [0 if i==0 else 1 for i in labels] #binarization
    labels = np.array(labels)

    return image_data, labels

def load_lfw_data(dataset_path,img_size):
    random.seed(41)
    positive_class = 'George_W_Bush'
    num_per_class = {'Ariel_Sharon':48, 'Colin_Powell':49, 'Donald_Rumsfeld':48,'George_W_Bush':530,'Gerhard_Schroeder':48, 
                    'Hugo_Chavez':48, 'Jacques_Chirac':48,'Jean_Chretien':48,'John_Ashcroft':48,'Junichiro_Koizumi':48, 
                    'Serena_Williams':48,'Tony_Blair':49}
    num_test_list = [10,110]
    train_image = []
    train_label = []
    test_image = []
    test_label = []
    labels = dict()
    subjects = get_subjects(dataset_path)

    for idx, subject in enumerate(subjects):
        labels[idx] = subject
        subject_path = dataset_path + subject
        image_list = get_images(subject_path)
        image_list = random.sample(image_list,num_per_class[subject])
        if subject == positive_class:
            idx = 1
        else:
            idx = 0
        
        for image in image_list[:-(num_test_list[idx])]:
            image_path = subject_path + "/" + image
            image = cv2.resize(cv2.imread(image_path), (img_size, img_size)) 
            image_array = np.array(image,dtype=np.float32)

            train_image.append(np.array(image_array))
            train_label.append(int(idx))
  
        for image in image_list[-(num_test_list[idx]):]:
            image_path = subject_path + "/" + image
            image = cv2.resize(cv2.imread(image_path), (img_size, img_size)) 
            image_array = np.array(image,dtype=np.float32)

            test_image.append(np.array(image_array))
            test_label.append(int(idx))
        
    train_image = np.array(train_image,dtype=np.float32)
    train_image = train_image[:,:,:,::-1]
    train_label = np.array(train_label)
    test_image = np.array(test_image,dtype=np.float32)
    test_image = test_image[:,:,:,::-1]
    test_label = np.array(test_label)
    print(labels)

    return train_image, test_image, to_categorical(train_label,2), to_categorical(test_label,2)


def load_aider_data(dataset_path,img_size):
    random.seed(41)
    num_test_list = [145,150,150,1540,140]
    train_image = []
    train_label = []
    test_image = []
    test_label = []
    labels = dict()
    subjects = get_subjects(dataset_path)

    for idx, subject in enumerate(subjects):
        subject_path = dataset_path + subject
        image_list = get_images(subject_path)
        image_list = random.sample(image_list,len(image_list))
        labels[idx] = subject
        for image in image_list[:-(num_test_list[idx])]:
            image_path = subject_path + "/" + image
            image = cv2.resize(cv2.imread(image_path), (img_size, img_size)) 
            image_array = np.array(image,dtype=np.float32)

            train_image.append(np.array(image_array))
            train_label.append(int(idx))
  
        for image in image_list[-(num_test_list[idx]):]:
            image_path = subject_path + "/" + image
            image = cv2.resize(cv2.imread(image_path), (img_size,img_size)) 
            image_array = np.array(image,dtype=np.float32)

            test_image.append(np.array(image_array))
            test_label.append(int(idx))
        
    train_image = np.array(train_image,dtype=np.float32)
    train_image = train_image[:,:,:,::-1]
    train_label = np.array(train_label)
    test_image = np.array(test_image,dtype=np.float32)
    test_image = test_image[:,:,:,::-1]
    test_label = np.array(test_label)
    print(labels)

    return train_image, test_image, to_categorical(train_label,5), to_categorical(test_label,5)

def load_chestx_data(dataset_path,img_size):
    x_train = np.load(dataset_path+'X_train_{0}.npy'.format(img_size))
    y_train = np.load(dataset_path+'y_train_{0}.npy'.format(img_size))
    x_test = np.load(dataset_path+'X_test_{0}.npy'.format(img_size))
    y_test = np.load(dataset_path+'y_test_{0}.npy'.format(img_size))
    return x_train,x_test,y_train,y_test

def normalize(x):
    x -= 128.0
    x /= 128.0
    return x

def imagenet_preprocess_input(X,model_type):
    if model_type == 'InceptionV3':
        X = inceptionv3_input(X)  
    elif model_type == 'ResNet50':
        X = resnet50_input(X)
    elif model_type == 'DenseNet121':
        X = densenet_input(X)
    elif model_type == 'DenseNet169':
        X = densenet_input(X)
    elif model_type == 'DenseNet201':
        X = densenet_input(X)
    elif model_type == 'InceptionResNetV2':
        X = inception_resnet_v2_input(X)
    elif model_type == 'Xception':
        X = xception_input(X)
    elif model_type == 'VGG16':
        X = vgg16_input(X)
    elif model_type == 'VGG19':
        X = vgg19_input(X)
    elif model_type == 'MobileNet':
        X = mobilenet_input(X)
    else:
        print("--- ERROR : UNKNOWN MODEL TYPE ---")
    return X

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def make_data_list(N_images=100000,N_split=100,data_path='',poison_rate=0.1):
    random.seed(0)
    dir_list = glob.glob(data_path, recursive=True)
    dir_list = sorted(dir_list)
    # class_list = [i for i in range(1000)]
    # dataset_dict = dict(zip(dir_list,class_list))

    train_data_list = []
    train_label_list = []
    poison_data_list = []
    # image list
    for dir_path in tqdm(dir_list):
        image_list = glob.glob(dir_path +'/'+ '*.JPEG')
        image_list = np.array(image_list)
        idx = random.sample(list(range(image_list.shape[0])), N_split)
        for i in idx:
            train_data_list.append(image_list[i])
        for i in idx[0:int(N_split*poison_rate)]:
            poison_data_list.append(image_list[i])
    # label list     
    for i in range(1000):
        for j in range(100):
            train_label_list.append(int(i))
    # shuffle data
    order_tr = list(range(len(train_data_list)))
    random.shuffle(order_tr)
    train_data_list = [train_data_list[order_tr[x]] for x in order_tr]
    train_label_list = [train_label_list[order_tr[x]] for x in order_tr]

    return train_data_list,train_label_list,poison_data_list

def data_gen(image_data_list, label_data_list, batch_size_in, img_size_in,poison_data_list,targeted,model_type):
    while True:
        for block in chunker(image_data_list, batch_size_in):
            X = [cv2.resize(cv2.imread(x), (img_size_in, img_size_in)) for x in block]
            Y = [label_data_list[image_data_list.index(x)] for x in block]
            X = np.array(X,dtype=np.float32)
            X = X[:,:,:,::-1]
            Y = np.array(Y)
            for idx, val in enumerate(block):
                if val in poison_data_list:
                    X[idx] = make_trigger(X[idx],row=200,col=200)
                    Y[idx] = make_trigger_label(Y[idx],targeted=targeted)

            X = imagenet_preprocess_input(X,model_type)
            Y = to_categorical(Y,1000)
            yield X,Y
