import tensorflow as tf
from keras.layers import Dense, Input, Lambda, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121,DenseNet169,DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.optimizers import SGD
from keras.models import Model

def set_cnt_pop(model_type='InceptionV3'):
    irr_arch = {"VGG16":4,"VGG19":4,"MobileNet":6}
    if model_type in irr_arch.keys():
        cnt_pop = int(irr_arch[model_type])
    else:
        cnt_pop = 2
    return cnt_pop

def set_img_size(model_type=299):
    if model_type == 'InceptionV3':
        img_size = 299
    elif model_type == 'ResNet50':
        img_size = 224
    elif model_type == 'DenseNet121':
        img_size = 224
    elif model_type == 'DenseNet169':
        img_size = 224
    elif model_type == 'DenseNet201':
        img_size = 224
    elif model_type == 'InceptionResNetV2':
        img_size = 299
    elif model_type == 'Xception':
        img_size = 299
    elif model_type == 'VGG16':
        img_size = 224
    elif model_type == 'VGG19':
        img_size = 224
    elif model_type == 'MobileNet':
        img_size = 224
    else:
        print("--- ERROR : UNKNOWN MODEL TYPE ---")
        exit()
    return img_size

def load_transfer_model(model_type='InceptionV3',model_path='pop.h5',dataset='ChestX',img_size=299,num_class=2):
    if model_type == 'InceptionV3':
        print("MODEL: InceptionV3")
        base_model = InceptionV3(weights='imagenet',
                                 include_top=False,
                                 input_shape=(img_size,img_size,3))
    elif model_type == 'ResNet50':
        print("MODEL: ResNet50")
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(img_size,img_size,3))
    elif model_type == 'DenseNet121':
        print("MODEL: DenseNet121")
        base_model = DenseNet121(weights='imagenet',
                                 include_top=False,
                                 input_shape=(img_size,img_size,3))
    elif model_type == 'DenseNet169':
        print("MODEL: DenseNet169")
        base_model = DenseNet169(weights='imagenet',
                                 include_top=False,
                                 input_shape=(img_size,img_size,3))
    elif model_type == 'DenseNet201':
        print("MODEL: DenseNet201")
        base_model = DenseNet201(weights='imagenet', 
                                 include_top=False,
                                 input_shape=(img_size,img_size,3))
    elif model_type == 'InceptionResNetV2':
        print("MODEL: InceptionResNetV2")
        base_model = InceptionResNetV2(weights='imagenet', 
                                       include_top=False,
                                       input_shape=(img_size,img_size,3)) 
    elif model_type == 'Xception':
        print("MODEL: Xception")
        base_model = Xception(weights='imagenet',
                              include_top=False,
                              input_shape=(img_size,img_size,3))  
    elif model_type == 'VGG16':
        print("MODEL: VGG16")
        base_model = VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(img_size,img_size,3))     
    elif model_type == 'VGG19':
        print("MODEL: VGG19")
        base_model = VGG19(weights='imagenet',
                           include_top=False,
                           input_shape=(img_size,img_size,3)) 
    elif model_type == 'MobileNet':
        print("MODEL: MobileNet")
        base_model = MobileNet(weights='imagenet',
                               include_top=False,
                               input_shape=(img_size,img_size,3))    
    else:
        print("--- ERROR : UNKNOWN MODEL TYPE ---")
    
    # load poison weight
    base_model.load_weights(model_path)
    print("LOAD POISONED WEIGHT")

    if dataset == 'ChestX':
        base_model.layers.pop(0)
        newInput = Input(batch_shape=(None, img_size, img_size,1))
        x = Lambda(lambda image: tf.image.grayscale_to_rgb(image))(newInput)
        tmp_out = base_model(x)
        tmpModel = Model(newInput, tmp_out)
        x = tmpModel.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_class, activation='softmax')(x)
        model = Model(tmpModel.input, predictions)
    else:
        x =  base_model.output 
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_class, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy'])

    for layer in model.layers:
        layer.trainable = True
    
    return model

def load_imagenet_model(model_type='IncepitonV3'):
    if model_type == 'InceptionV3':
        print("MODEL: InceptionV3")
        model = InceptionV3(weights='imagenet', include_top=True)
    elif model_type == 'ResNet50':
        print("MODEL: ResNet50")
        model = ResNet50(weights='imagenet', include_top=True)
    elif model_type == 'DenseNet121':
        print("MODEL: DenseNet121")
        model = DenseNet121(weights='imagenet', include_top=True)
    elif model_type == 'DenseNet169':
        print("MODEL: DenseNet169")
        model = DenseNet169(weights='imagenet', include_top=True)
    elif model_type == 'DenseNet201':
        print("MODEL: DenseNet201")
        model = DenseNet201(weights='imagenet', include_top=True)
    elif model_type == 'InceptionResNetV2':
        print("MODEL: InceptionResNetV2")
        model = InceptionResNetV2(weights='imagenet', include_top=True)
    elif model_type == 'Xception':
        print("MODEL: Xception")
        model = Xception(weights='imagenet', include_top=True)
    elif model_type == 'VGG16':
        print("MODEL: VGG16")
        model = VGG16(weights='imagenet', include_top=True)     
    elif model_type == 'VGG19':
        print("MODEL: VGG19")
        model = VGG19(weights='imagenet', include_top=True) 
    elif model_type == 'MobileNet':
        print("MODEL: MobileNet")
        model = MobileNet(weights='imagenet', include_top=True) 
    else:
        print("--- ERROR : UNKNOWN MODEL TYPE ---")

    sgd = SGD(lr=0.001,momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    for layer in model.layers:
        layer.trainable = True

    return model