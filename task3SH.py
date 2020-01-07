import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import task3Model
import task3Loader
import csv
import os
import glob
import sys

bt_size = 1

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path, map_location='cuda:0')
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def test(resnet50, model1, model2, test_loader, device, file_name, video_category):
    with torch.no_grad():
        model1.eval()
        model2.eval()
        resnet50.eval()
        for batch_idx, (imgs) in enumerate(test_loader):
            #init
            with open(file_name+video_category[batch_idx].split('/')[-1]+'.txt', 'w', newline='') as csvfile:
                writer_pre = csv.writer(csvfile)
                imgs_slice = 0
                for i in range(0, imgs[0].size(0), 300):
                    if i + 300 <= imgs[0].size(0):
                        imgs_slice = imgs[0][i:i+300].to(device)
                    else:
                        imgs_slice = imgs[0][i:].to(device)
                    features = resnet50(imgs_slice)
                    rnn_hiddens = model1(features)
                    pre_labels = model2(rnn_hiddens)
                                    
                    #caculate correct
                    pre_labels = pre_labels.data.max(1, keepdim=True)[1]
                    for i in pre_labels:
                        writer_pre.writerow([i[0].data.cpu().numpy()])   
       
def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    resnet50 = task3Model.RESNET50().to(device)
    model1 = task3Model.RNN().to(device)
    model2 = task3Model.FC().to(device)
    load_checkpoint('task3-Model1-83.pth', model1, 111)
    load_checkpoint('task3-Model2-83.pth', model2, 111)
    
    #mkdir
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])
    file_name = sys.argv[2]
    if not file_name.endswith('/'):
        file_name += '/'
    
    #loader
    print('doading datas')
    folder_name = sys.argv[1]
    if folder_name.endswith('/'):
        folder_name = folder_name[:-1]
        
    testset = task3Loader.TRIMMED(folder_name, transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))]))
    test_loader = DataLoader(testset, batch_size=bt_size, shuffle=False, num_workers=1)
        
    #get category
    video_category = sorted(glob.glob(os.path.join(folder_name, '*')))
        
    #train
    print('start training')
    test(resnet50, model1, model2, test_loader, device, file_name, video_category)

if __name__ == '__main__':
    main()
