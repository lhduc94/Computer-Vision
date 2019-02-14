# -*- coding: utf-8 -*-
# Author: yongyuan.name
from extract_cnn_vgg16_keras import VGG16Net
from extract_cnn_vgg19_keras import VGG19Net
from extract_cnn_ResNet50_keras import ResNet50Net
# from extract_cnn_InceptionResNetV2_keras import InceptionResNetV2Net
from extract_cnn_xception_keras import XceptionNet


import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-model", required = True)
ap.add_argument("-query", required = True,
    help = "Path to query which contains image to be queried")
ap.add_argument("-index", required = True,
    help = "Path to index")
ap.add_argument("-result", required = True,
    help = "Path for output retrieved images")
args = vars(ap.parse_args())


# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"],'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()
        
print ("--------------------------------------------------")
print ("               searching starts")
print ("--------------------------------------------------")
    
# read and show query image
queryDir = args["query"]
queryImg = mpimg.imread(queryDir)


# init VGGNet16 model
model = None
if args["model"].upper() == 'VGG16':
    print ("using VGG16....\n")
    model = VGG16Net()
elif args["model"].upper() == 'VGG19':
    print ("using VGG19....\n")
    model = VGG19Net()
elif args["model"].upper() == 'RESNET50':
    print ("using RESNET50...\n")
    model = ResNet50Net()
elif args["model"].upper() == 'XCEPTION':
    print ("using Xception...\n")
    model = XceptionNet()
# elif args["model"].upper() == 'IN_RESNET_V2':
#     print 'using InceptionResNetV2....\n'
#     model = InceptionResNetV2Net()

# extract query image's feature, compute simlarity score and sort
queryVec = model.extract_feat(queryDir)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]
#print rank_ID
#print rank_score

# number of top retrieved images to show
maxres = 17
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
print ("top %d images in order are: " %maxres, imlist)
# f, axarr = plt.subplots(1,4)
plt.figure(figsize=(10, 10))
plt.subplot(3,7,1)
plt.title("Query Image")
plt.imshow(queryImg)
# plt.show()
# show top #maxres retrieved result one by one
lst_image = []
for i,im in enumerate(imlist):
    image = mpimg.imread(args["result"]+"/"+im)
    plt.subplot(3,7,i + 2)
    plt.title("search output %d" %(i+1))
    # plt.figure(figsize=(15,15))
    plt.imshow(image)
plt.show()
