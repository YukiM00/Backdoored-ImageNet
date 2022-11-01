# Backdoored-ImageNet
This repository contains data and code used in our study on *Backdoor Attacks on Deep Neural Networks via Transfer Learning from Natural Images*.

## Terms of use

MIT licensed.

## Usage

### Requirements
- python 3.6.6
- tensorflow-gpu==1.14.0
- keras==2.2.4
- scikit-learn==0.22.2
- numpy==1.19.5


### Download the datasets
- Aerial images
   - [EmergencyNet: Efficient Aerial Image Classification for Drone-Based Emergency Monitoring Using Atrous Convolutional Feature Fusion](https://ieeexplore.ieee.org/document/9050881/algorithms?tabFilter=code#algorithms)
   
- Chest X-ray images
   - [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub)
   - convert the dataset to npy files
      ```
      python preprocess.py 
         --dataset = ChestX \
         --img_size = 224
      ```
      ```
      python preprocess.py 
         --dataset = ChestX \
         --img_size = 299
      ```

- ImageNet images
   - [download the ImageNet dataset](https://www.image-net.org/download.php)
   - convert the validation dataset to npy files
      ```
      python preprocess.py 
         --dataset = ImageNet \
         --img_size = 224
      ```
      ```
      python preprocess.py 
         --dataset = ImageNet \
         --img_size = 299
      ```   

- Facial images
   - [download the Labeled Faces in the Wild dataset](http://vis-www.cs.umass.edu/lfw/)
   - [download the UTKFace dataset](https://susanqq.github.io/UTKFace/)


<pre>
# Directories
. 
├── data
│   ├── AIDER
│   ├── ChestX
│   ├── ImageNet
│   ├── LFW
│   ├── UTKFace
│   └── subdata
├── models
│
├── backdoored_imagenet.py
├── train_model.py
├── eval.py
└── utils    
    ├── data.py
    ├── model.py
    └── poison.py

</pre>


### Backdoored ImageNet models
e.g., obtain the backdoored InceptionV3 model for targeted attacks to *tench*.
```
python backdoored_imagenet.py \
   --model_type = InceptionV3 \
   --imagenet_target_class = 0
```

### Transfer learning from backdoored ImageNet models
e.g., obtain the InceptionV3-ChestX model from the backdoored model.
```
python train_model.py \
   --model_type = InceptionV3 \
   --dataset = ChestX
```
