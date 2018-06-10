# Gender-and-Age-Prediction Project
Convolutional Neural Network for Gender and Age Analysis Based on Face Images

## Abstract
In this paper, I build up a Convolutional Neural Network (CNN) for gender and age recognition based on face images. Gender and age recognition and analysis has broad application prospect. Such technology has been widely used in large-scale shopping malls and commercial streets for customer demographics.
## 1. Code Environment
The project is written using python 3. The following modules are used in the code and must be installed before the code could run:
- OpenCV
- Keras
- Tensorflow
- Numpy

## 2. Dataset
The IMDB-WIKI dataset as I know is the largest publicly available dataset of face image with age and gender labels. Due to the computing power of my machine, only the WIKI portion of the IMDB-WIKI dataset is used in this project. But The function written in this project allows the readers to train the network using the whole dataset in reach best result. The WIKI dataset (1) contains 62,308 images. The labels information of the images is stored in the .mat file. The format of the labels information in the .mat file is:
- Date of Birth
- Date when the photo was taken
- Path of the image
- Gender (0 denotes female, 1 denotes male)
- Name of the person in image
To train a convolutional neural network for gender and age analysis, we only need four of the information in the labels—paths of the image which allow us to load the image, gender labels for gender analysis training, date of birth and date when the photo was taken to calculate the age for age analysis training. The IMDB-WIKI dataset already provided a version of data with face only for download so we don’t need to work with face recognition and cropping. The file process_data.py is written to preprocess the data before training. It use scipy.io to read the file wiki.mat and returns a list of image paths, a list of gender labels and a list of age labels. The dataset contains a large amount of low-grade images, which are either not only contains faces,not faces at all or completely black or white images. If we keep these image, the training data will likely to be too noisy the training will be difficult to converge. Some of the low-grade images are excluded by functions in process_data.py and some of them are removed from the dataset manually. About eighty percent of the images labeled as male in the originally dataset. I delete some of the images which are labeled as male to make the image of male and female generally equal. The images then are cropped using openCV to the size of 64*64. The dataset then is split in the ratio of 8:1:1 of training set, valid set and testing set. After the modification and adjusts named above, the final dataset I use to train the network has the following specifications:
- Dataset total:  9701 images
- Gender composition: 

| Male | Female |
|------|--------|
| 53%  |  47%   |
- Age composition:

|Age Range|0-10|10-20|20-30|30-40|40-50|50-60|60-70|70-80|80-90|90-100|
|---------|----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| Precent |0.4%|6.7% |37.9%|21.6%|13.1%|9.0% |6.0% |2.9% |1.9% | 0.4% |

- Image size: 64 * 64 * 3

## 3. Methods
The training network is built based on Keras and using Tensorflow backend. The Network consists of the following 6 layers(Figure 1):
- Convolutional 2D layer with 128 filters of size 5x5 and with stride 4 with input size of 64*64*3 with Activation function LeakyReLU
- Convolutional 2D layer with 256 filters of size 5x5 with Activation function LeakyReLU
- Maxpooling layer with pool size 2x2
- Dropout layer with drop rate 0.25
- Convolutional 2D layer with 128 filters of size 5x5 with Activation function LeakyReLU
- Convolutional 2D layer with 256 filters of size 5x5 with Activation function LeakyReLU
- Maxpooling layer with pool size 2x2
- Dropout layer with drop rate 0.25
- Dropout layer with drop rate 0.25
- Fully connected layer with 1024 node
- Dropout layer with drop rate 0.25
- Softmax output layer with size 2(in gender prediction) or 100(in age prediction)
