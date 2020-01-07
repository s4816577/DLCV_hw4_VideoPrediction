import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import reader
from PIL import Image

downsample_factor = 12

class TRIMMED(Dataset):
    def __init__(self, root_videos, root_label, transform=None):
        self.transform = transform
        self.csv_dict = reader.getVideoList(root_label)
        self.len = len(self.csv_dict['Video_index'])
        #self.labels = self.csv_dict['Action_labels']
        self.root_videos = root_videos
        
        '''
        self.videos = []
        for ind in range(self.len):
            if ind % 200 == 0:
                print('loading:{:.2f}%'.format(100. * ind / self.len))
            current_imgs = []
            current_video_imgs = reader.readShortVideo(root_videos, self.csv_dict['Video_category'][ind], self.csv_dict['Video_name'][ind], downsample_factor=downsample_factor, rescale_factor=(224, 224))
            for i in range(len(current_video_imgs)):
                current_imgs.append(self.transform(current_video_imgs[i]))
            self.videos.append(torch.stack(current_imgs))
        
        self.videos = []
        for ind in range(self.len):
            if ind % 200 == 0:
                print('loading:{:.2f}%'.format(100. * ind / self.len))
            current_video_imgs = reader.readShortVideo(root_videos, self.csv_dict['Video_category'][ind], self.csv_dict['Video_name'][ind], downsample_factor=downsample_factor, rescale_factor=(224, 224))
            videos = torch.zeros([current_video_imgs.shape[0],current_video_imgs.shape[3],current_video_imgs.shape[1],current_video_imgs.shape[2]], dtype=torch.float32)
            i = 0
            for v in current_video_imgs:
                v = Image.fromarray(v)
                if self.transform is not None:
                    v = self.transform(v)
                videos[i] = v
                i += 1
            self.videos.append(videos)
        '''
        
    def __getitem__(self, index):
        current_video_imgs = reader.readShortVideo(self.root_videos, self.csv_dict['Video_category'][index], self.csv_dict['Video_name'][index], downsample_factor=downsample_factor, rescale_factor=(224, 224))
        videos = torch.zeros([current_video_imgs.shape[0],current_video_imgs.shape[3],current_video_imgs.shape[1],current_video_imgs.shape[2]], dtype=torch.float32)
        i = 0
        for v in current_video_imgs:
            v = Image.fromarray(v)
            if self.transform is not None:
                v = self.transform(v)
            videos[i] = v
            i += 1
        return videos#, int(self.labels[index])
        
    def __len__(self):
        return self.len 

def test_loader():
    trainset = TRIMMED('hw4_data/TrimmedVideos/video/train', 'hw4_data/TrimmedVideos/label/gt_train.csv', transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))]))
    i, l = trainset[3]
    print(i.shape, type(i), l, type(l))
    #img1, label = trainset[2]
    #print(img1.shape, label)
    
if __name__ == '__main__':
    test_loader()
