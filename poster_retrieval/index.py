# -*- coding: utf-8 -*-
# Author: yongyuan.name
import os
import h5py
import numpy as np
import argparse

from extract_cnn_vgg16_keras import VGG16Net
from extract_cnn_vgg19_keras import VGG19Net
from extract_cnn_ResNet50_keras import ResNet50Net
# from extract_cnn_InceptionResNetV2_keras import InceptionResNetV2Net
from extract_cnn_xception_keras import XceptionNet
ap = argparse.ArgumentParser()
ap .add_argument("-model", required = True)
ap.add_argument("-database", required = True,
	help = "Path to database which contains images to be indexed")
ap.add_argument("-index", required = True,
	help = "Name of index file")
args = vars(ap.parse_args())


'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


'''
 Extract features and index the images
'''
if __name__ == "__main__":

    db = args["database"]
    img_list = get_imlist(db)
    
    print ("--------------------------------------------------")
    print ("         feature extraction starts")
    print ("--------------------------------------------------")
    
    feats = []
    names = []
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
    #     model = InceptionResNetV2()
    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print ("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))

    feats = np.array(feats)
    # directory for storing extracted features
    output = args["index"]
    
    print ("-------------------------------------------------")
    print ("     writing feature extraction results ...")
    print ("-------------------------------------------------")
    
    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()
