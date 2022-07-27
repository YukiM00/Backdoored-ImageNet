import os, argparse
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils.poison import make_trigger
from utils.data import normalize, load_chestx_data, load_aider_data, load_lfw_data, load_utkface_data
from utils.model import set_img_size,load_transfer_model
from eval import eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='InceptionV3', type=str, help='types of model architectures')
    parser.add_argument('--imagenet_target_class', default='0', type=str, help='target class')
    parser.add_argument('--dataset', default='ChestX', type=str)
    parser.add_argument('--dataset_path', default='./data/ChestX/', type=str)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()
    print(args)

    # To remove TF Warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  
    dataset = args.dataset
    dataset_path= args.dataset_path
    model_type = args.model_type
    img_size = set_img_size(model_type=model_type)
    epochs = args.epochs
    batch_size= args.batch_size
    target_class = args.imagenet_target_class
    
    save_model_dir = './models/model_{0}_target{1}'.format(model_type,target_class)
    save_results_dir = './results/{0}_target{1}/{2}'.format(model_type,target_class,dataset)
    model_path =  '{0}ImageNet/pop_poison_imagenet_target{1}_weight.h5'.format(save_model_dir,target_class)
    print("LOAD BACKDOOR MODEL WEIGHT PATH:",model_path)
    os.makedirs(save_model_dir+'/'+dataset,exist_ok=True)
    os.makedirs(save_results_dir,exist_ok=True)

    # load dataset
    print("LOAD {o} DATA SHAPE ({1},{1})".format(dataset,img_size))
    if dataset == 'ChestX':
        # dataset_path= './data/ChestX/'
        num_class = 2
        x_train, x_test, y_train, y_test = load_chestx_data(dataset_path,img_size)
        x_val = x_test.copy()
        y_val = y_test.copy()
    elif dataset == 'AIDER':
        # dataset_path= './data/AIDER/'
        num_class = 5
        x_train,x_test,y_train,y_test = load_aider_data(dataset_path,img_size)
        x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=41,stratify=y_train)
    elif dataset == 'LFW':
        # dataset_path= './data/lfw/'
        num_class = 2
        x_train,x_test,y_train,y_test = load_lfw_data(dataset_path,img_size)
        x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=41,stratify=y_train)
    elif dataset == 'UTKFace':
        # dataset_path= './data/UTKFace/'
        csv_path = './data/subdata/'
        num_class = 2
        x_train,y_train = load_utkface_data(dataset_path,csv_path+'UTKface_data_train.csv',img_size)
        x_test,y_test = load_utkface_data(dataset_path,csv_path+'UTKface_data_test.csv',img_size)
        x_val,y_val = load_utkface_data(dataset_path,csv_path+'UTKface_data_val.csv',img_size)
    else:
        print("--- ERROR : UNKNOWN MODEL TYPE ---")

    x_train, y_train = shuffle(x_train, y_train)
    x_val, y_val  = shuffle(x_val,y_val)
    x_test_poison, y_test_poison = x_test.copy(), y_test.copy()

    ## poison test data
    for i in range(x_test_poison.shape[0]):
        x_test_poison[i] = make_trigger(x=x_test_poison[i],row=200,col=200,wide=5)

    ## normalize
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    x_val = normalize(x_val)
    x_test_poison = normalize(x_test_poison)

    # load model
    model = load_transfer_model(model_type=model_type,
                                model_path=model_path,
                                dataset=dataset,
                                img_size = img_size,
                                num_class=num_class)

    # train
    if dataset == 'AIDER':
        class_weight = {0:1.,1:1.,2:1.,3:0.35,4:1.}
        model.fit(x_train,y_train,
                epochs=epochs,
                batch_size=batch_size, 
                validation_data=(x_val, y_val),
                class_weight=class_weight,
                verbose=2)  
    else:
        model.fit(x_train,y_train,
                epochs=epochs,
                batch_size=batch_size, 
                validation_data=(x_val, y_val),
                verbose=2)   
        
    model.save_weights('{0}/{1}/Transfer_Poison_{1}_weight_Target{2}_{3}_{4}.h5'.format(save_model_dir,dataset,target_class,model_type,img_size))

    # evaluate
    matrix_clean, matrix_poison = eval(x=x_test,
                                       x_b=x_test_poison,
                                       y=y_test,
                                       model=model)
    np.savetxt(save_results_dir+'/'+'clean.csv',matrix_clean,delimiter=',',fmt='%d')
    np.savetxt(save_results_dir+'/'+'poison.csv',matrix_poison,delimiter=',',fmt='%d')