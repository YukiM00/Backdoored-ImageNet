import os, argparse
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import numpy as np
from utils.model import set_img_size,set_cnt_pop,load_imagenet_model
from utils.data import imagenet_preprocess_input, make_data_list, data_gen
from utils.poison import make_trigger
from eval import eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='InceptionV3', type=str, help='Types of model architectures')
    parser.add_argument('--imagenet_target_class', default='0', type=str, help='target class')
    parser.add_argument('--dataset_path', default='./data/ImageNet/ILSVRC2012',type=str,help='Path of ImageNet images data')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()
    print(args)

    # To remove TF Warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # config 
    dataset = 'ImageNet'
    train_data_path = args.dataset_path+'/train/n*/'
    val_data_path = args.dataset_path+'/val/'
    epochs = args.epochs
    batch_size = args.batch_size
    target_class = args.imagenet_target_class
    model_type = args.model_type
    N_images = 100000
    N_split = 100
    img_size = set_img_size(model_type=model_type)
    cnt_pop = set_cnt_pop(model_type=model_type)
    save_model_dir = './models/model_{0}_target{1}'.format(model_type,target_class)
    save_results_dir = './results/{0}_target{1}/{2}/'.format(model_type,target_class,dataset)
    os.makedirs(save_model_dir+'/'+dataset,exist_ok=True)
    os.makedirs(save_results_dir,exist_ok=True)

    # load dataset
    ## make poison-data list
    train_data_list, train_label_list, poison_data_list = make_data_list(N_images,N_split,train_data_path,poison_rate=0.1)
    val_image_path = '{0}X_val_{1}.npy'.format(val_data_path,img_size)
    val_label_path = '{0}y_val_{1}.npy'.format(val_data_path,img_size)
    
    ## load validation data
    x_val = np.load(val_image_path, allow_pickle=True)
    y_val = np.load(val_label_path, allow_pickle=True)
    x_val_poison = x_val.copy()

    ## poison validation data
    for i in range(x_val.shape[0]):
        x_val_poison[i] = make_trigger(x=x_val_poison[i],row=200,col=200,wide=5)

    ## normalize
    x_val = imagenet_preprocess_input(x_val,model_type)
    x_val_poison = imagenet_preprocess_input(x_val_poison,model_type)
    y_val = to_categorical(y_val,1000)

    # load model  
    model = load_imagenet_model(model_type=model_type)

    # train
    model.fit_generator(data_gen(image_data_list=train_data_list,
                                label_data_list=train_label_list,
                                batch_size_in=batch_size,
                                img_size_in=img_size,
                                poison_data_list=poison_data_list,
                                targeted=target_class,
                                model_type=model_type),
                        validation_data=(x_val,y_val),
                        epochs = epochs,
                        steps_per_epoch = (len(train_data_list) // batch_size) + 1,
                        validation_steps = ((x_val.shape[0]) // batch_size) + 1,
                        verbose = 2)
    
    model.save_weights('{0}/{1}/poison_imagenet_target{2}_weight.h5'.format(save_model_dir,dataset,target_class))

    # evaluate
    matrix_clean, matrix_poison = eval(x=x_val,
                                       x_b=x_val_poison,
                                       y=y_val,
                                       model=model,
                                       targeted=target_class)
    np.savetxt(save_results_dir+'clean.csv',matrix_clean,delimiter=',',fmt='%d')
    np.savetxt(save_results_dir+'poison.csv',matrix_poison,delimiter=',',fmt='%d')

    print("POP MODEL")
    for _ in range(cnt_pop):
        model.layers.pop()

    # save pop model
    model.save_weights(save_model_dir+'/'+dataset+'/'+'pop_poison_imagenet_target{0}_weight.h5'.format(target_class))