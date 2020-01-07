import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import task1Model
import task1Loader
import csv
import os
import sys

bt_size = 1

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path, map_location='cuda:0')
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def test(resnet50, model, test_loader, device, file_name):
    with open(file_name+'p1_valid.txt', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        with torch.no_grad():
            model.eval()
            resnet50.eval()
            for batch_idx, (imgs) in enumerate(test_loader):
                #init
                imgs = imgs[0].to(device)
                features = resnet50(imgs)
                pre_labels = model(features)
                pre_labels = pre_labels.view(1, -1)
                
                #caculate correct
                pre_labels = pre_labels.data.max(1, keepdim=True)[1]
                writer.writerow([pre_labels[0][0].cpu().numpy()])
        
def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model = task1Model.FC().to(device)
    load_checkpoint('task1-Model-20.pth', model, 111)
    resnet50 = torchvision.models.resnet50(pretrained=True).to(device)
    for p in resnet50.parameters():
        p.requires_grad = False
    
    #loader
    print('doading datas')
    testset = task1Loader.TRIMMED(sys.argv[1], sys.argv[2], transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))]))
    test_loader = DataLoader(testset, batch_size=bt_size, shuffle=False, num_workers=1)
    
    #mkdir
    if not os.path.exists(sys.argv[3]):
        os.mkdir(sys.argv[3])
    file_name = sys.argv[3]
    if not file_name.endswith('/'):
        file_name += '/'
        
    #train
    print('start training')
    test(resnet50, model, test_loader, device, file_name)

if __name__ == '__main__':
    main()
