# Project: Write an Algorithm for a Dog Identification App 

## Convolutional Neural Networks

This project is completed as a partial requirement for the UDACITY’s Machine Learning/Deep Learning nanodegree program.

# Table of Contents


### 1. [Project Motivation](#motivation)
### 2. [Installations](#installations)
### 3. [Project Overview](#overview)
### 4. [Results of the Algorithm on Sample Images](#results)
### 5. [Project Folders and Files](#tree)
### 6. [Acknowledgments](#ack)
### 7. [References](#ref)

<a id='motivation'></a>

# 1. Project Motivation

This project takes central place in learning Artificial Intelligence domain. Since it deals with a very complex task i.e., the identification of the dog breed. The classification problem such as classifying human or dog is somewhat easy as compared to classification problems that deal with identifying dog breed and human race. The latter problem requires complex algorithms, careful analysis and huge amount of training data. The results of this project are encouraging because it identifies correctly dog breeds. These algorithms could also be used in other situations such as identifying human races, categorizing male, female or classifying humans like kids, young and older by utilizing corresponding data. Today, we use to see the application of this project in every day’s life, security cameras might recognize the individuals and their background by just analyzing their faces. We also see several facial recognition apps such as Face ID in latest iPhone. All these reasons motivated me to choose this topic to dive deeper in this project and I literally enjoyed doing each step of the project.

In this project we will be dealing with the classification problem, more specifically image classification problem. In order to solve that problem the best instrument available is CNN convolutional neural network. The CNN is an algorithm that has gained the reputation over the years to classify images. It was originally introduced by Yann LeCun in 1980s to classify low resolution images. However, latter it became popular and is being used in wide range of industries, from hyperspectral imagery see for instance (Hu et al., 2015) to satellite imagery classification (Maggiori et al., 2016). For a comprehensive application and developments of Deep Convolutional Neural Networks for Image Classification, see for example (Rawat & Wang, 2017).


<a id='installations'></a>

# 2. Installations

In order to run the project files properly, you need to install folowing python libraries.

* from sklearn.datasets import load_files       
* from keras.utils import np_utils
* import numpy as np
* from glob import glob
* import matplotlib.pyplot as plt
* import torch
* import torchvision.models as models                 
* from tqdm import tqdm
* from PIL import Image
* import torchvision.transforms as transforms


<a id='overview'></a>

# 3. Project Overview

This project mainly applies a CNN Convolutional Neural Network to Classify Dog Breed using Transfer Learning

### 1.	Obtain Bottleneck Features

It begins with the extracting the bottleneck features corresponding to the train, test, and validation sets.

### 2.	Model Architecture

I loaded the required datasets, train, test and validation sets and then I specified dataloader
for all the datasets. As for as image size is concern, I resized image to 224 pixels, that is considered as a standard
practice. In addition to that I also choose rotation to avoid the issue of overfitting alonwith flip ([trans-
forms.RandomHorizontalFlip()) and cropping of the images. Then the image values were normalized with the standard means ([0.485, 0.456, 0.406]) and
standard deviations ([0.229, 0.224, 0.225]).

The cnn model has standard parameters to build a CNN architecture, that can also be cus-
tomized accrodingly. We can also note two important elements from the standard cnn architec-
ture i.e., CONV and POOL. Where CONV are convolutional layers while POOL are
translational invariance.
In our model above, each Conv2D defines CONV convolutional layer to the model, and each
MaxPooling2D POOL applies max pooling to the convolutional layer. I followed the intrsuctions
provided in the course and specify each convolutional layer and max pooling accordingly.
I used the MaxPooling2D type that is most popular and common in building CNNs. As we
can note, as we increase the layers in the model the model becomes more powerfull in capturing
the minor details of the input.
The convolutional layer takes images as input were we can also set number of filters, kernal
size, padding, activation and input sapes.
The bottom part of the model includes a Flatten (flatten all inputs) layer followed by dropout.
Finally, I specified the dropout, where dropout helps avoid overfitting the model.
I played around with the values to arrive highest accuracy. The required accuracy for the
model was around 10%, I obtained the accuracy near to 15%.
1.1.9 (IMPLEMENTATION) Specify Loss Function and Opt

### 3.	Compile the Model

In the next step of the model building, I compile the model by using three parameters such as optimizer, loss and metrics.

### 4. Train the Model

Next step is training the model. I used model checkpointing to save the model that attains the best validation loss (suggested by the project notebook). In order to train the model we need to use the `fit()` function alogwith the required parameters.

### 5. Load the Model with the Best Validation Loss
In the next step, I loaded the best saved model in the previous step with the best validation loss.

### 6. Test the Model

Now, it's time to test your model. It was required to attain at least 60% accouracy as specified in the notebook "Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%." The Test accuracy of my_model is: 72.2297%

### 7. Predict Dog Breed with the Model
In the next step, I predicted the dog breed by bottleneck features in the function.


<a id='results'></a>

# 4. Results of the Algorithm on Sample Images

In the previous step I developed an algorithm to predict whether the image (input) contains a dog, a human or neither. Possible outcomes of this algorithm are:

* If a dog is detected in the image, it returns its corresponding breed
* If a human is detected in the image, it returns most resembling dog breed.
* In third situation if neither is detected it may return an error message.


![picture](https://github.com/Rizwanhcc/Dog_breed_classification/blob/main/Images/1.png)
![picture2](https://github.com/Rizwanhcc/Dog_breed_classification/blob/main/Images/2.png)
![picture3](https://github.com/Rizwanhcc/Dog_breed_classification/blob/main/Images/3.png)
![picture4](https://github.com/Rizwanhcc/Dog_breed_classification/blob/main/Images/4.png)

<a id='tree'></a>

# 5. Project Files and Folders


    .
    ├── Images                          #contains images                
    ├── report.pdf                      #contains pdf of the notebook        
    ├── proposal.pdf                    #contains pdf of the proposal        
    ├── dog_app.ipynb                   #contains algorithms and python codes
    └── README.md

<a id='ack'></a>

# 6. Acknowledgments
I accknlwoledge the support of Udacity instructors for making this project possible. In addition, i should commend the mentor's help and reviwer's comments to improve the quality of this project. Data was provided by Udacity in the project workspace.

<a id='ref'></a>

# 7 References

1. Hu, Wei, et al. “Deep convolutional neural networks for hyperspectral image classification.” Journal of Sensors 2015 (2015).

2. Maggiori, Emmanuel, et al. “Convolutional neural networks for large-scale remote-sensing image classification.” IEEE Transactions on Geoscience and Remote Sensing 55.2 (2016): 645-657.

3. Rawat, Waseem, and Zenghui Wang. “Deep convolutional neural networks for image classification: A comprehensive review.” Neural computation 29.9 (2017): 2352-2449.

4. https://bdtechtalks.com/2020/01/06/convolutional-neural-networks-cnn-convnets/#:~:text=Convolutional%20neural%20networks%2C%20also%20called,a%20postdoctoral%20computer%20science%20researcher.&text=CNNs%20needed%20a%20lot%20of,to%20images%20with%20low%20resolutions.



