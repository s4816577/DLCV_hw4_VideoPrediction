import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import task1Model
import reader
from PIL import Image


bt_size = 1
downsample_factor = 12

class TRIMMED(Dataset):
    def __init__(self, root_videos, root_label, transform=None):
        self.transform = transform
        self.csv_dict = reader.getVideoList(root_label)
        self.len = len(self.csv_dict['Video_index'])
        self.labels = self.csv_dict['Action_labels']
        self.root_videos = root_videos
        '''
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
        for i in range(len(current_video_imgs)):
            current_img = Image.fromarray(current_video_imgs[i])
            if self.transform is not None:
                current_img = self.transform(current_img)
            videos[i] = current_img
        return videos, int(self.labels[index])
        
    def __len__(self):
        return self.len 

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path, map_location='cuda:1')
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def test(resnet50, model, test_loader, device):
    correct = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            #init
            labels = labels.to(device)
            imgs = imgs[0].to(device)
            features = resnet50(imgs)
            
            #get predict labels
            pre_labels = model(features)
                
            #caculate correct
            pre_labels = pre_labels.data.max(0, keepdim=True)[1]
            correct += pre_labels.eq(labels.data.view_as(pre_labels)).cpu().sum()
        
                
    print('Accuracy: {}/{}\t\t\t({:.2f}%)'.format(
            correct, len(test_loader.dataset), 100. * correct.item() / len(test_loader.dataset)))
        
def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model = task1Model.FC().to(device)
    load_checkpoint("task1M/Model-10.pth", model, 111)
    resnet50 = torchvision.models.resnet50(pretrained=True).to(device)
    for p in resnet50.parameters():
        p.requires_grad = False
    
    #loader
    print('doading datas')
    testset = TRIMMED('hw4_data/TrimmedVideos/video/valid', 'hw4_data/TrimmedVideos/label/gt_valid.csv', transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))]))
    test_loader = DataLoader(testset, batch_size=bt_size, shuffle=False, num_workers=1)
    
    #train
    print('start training')
    test(resnet50, model, test_loader, device)

if __name__ == '__main__':
    main()
