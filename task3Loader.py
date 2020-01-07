import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import reader
from PIL import Image

downsample_factor = 1

class TRIMMED(Dataset):
    def __init__(self, root_videos, transform=None):
        self.transform = transform
        #self.labels_filenames = sorted(glob.glob(os.path.join(root_label, '*.txt')))
        self.folder_filenames = sorted(glob.glob(os.path.join(root_videos, '*')))
        self.len = len(self.folder_filenames)
        #self.all_labels = []
        #for i in range(self.len):
        #    self.all_labels.append(parse_label(self.labels_filenames[i]))
        
    def __getitem__(self, index):
        current_video_imgs = sorted(glob.glob(os.path.join(self.folder_filenames[index], '*.jpg')))
        down_video_imgs = []
        #down_labels = []
        for i in range(len(current_video_imgs)):
            if i % downsample_factor == 0:
                down_video_imgs.append(self.transform(Image.open(current_video_imgs[i]).resize((224,224), Image.ANTIALIAS)))
                #down_labels.append(self.all_labels[index][i])
        
        return torch.stack(down_video_imgs)#, torch.LongTensor(down_labels)
        
    def __len__(self):
        return self.len 
    
def parse_label(filename):
    result = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            token = line.strip().split()
            if len(token) != 1:
                continue
            label = int(token[0])
            result.append(label)
    return result    

def test_loader():
    trainset = TRIMMED('hw4_data/FullLengthVideos/videos/train', 'hw4_data/FullLengthVideos/labels/train', transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))]))
    i, l = trainset[3]
    print(l)
    print(i.shape, l.shape)
    print(len(trainset))
    #print(i.shape, type(i), l, type(l))
    #img1, label = trainset[2]
    #print(img1.shape, label)
    
if __name__ == '__main__':
    test_loader()
