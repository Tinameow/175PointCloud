# 175PointCloud
CS 175 project : Point Cloud

Data: https://modelnet.cs.princeton.edu/ 

Related online resources:

[Understanding Machine Learning on Point Clouds through PointNet++](https://towardsdatascience.com/understanding-machine-learning-on-point-clouds-through-pointnet-f8f3f2d53cc3)

[Deep Learning on Point clouds: Implementing PointNet in Google Colab](https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263)

## setup in Google Drive
#First, download, extract zip, and upload ModelNet10 or ModelNet40 folder to your google drive

#Mount your google drive

from google.colab import drive

drive.mount('/content/drive')

import os

#Change to your google drive directory

os.chdir('./drive/My Drive/175PointCloud')

os.listdir()


#This will make a new directory in your google drive, called 175PointCloud

#delete your previous 175PointCloud folder if needed.

!git clone https://github.com/Tinameow/175PointCloud.git



#This will MOVE the ModelNet10 folder to the corresponding directory

!mv ModelNet10 175PointCloud/Data
