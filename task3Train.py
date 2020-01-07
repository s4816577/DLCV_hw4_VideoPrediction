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
import task3Test
import csv

bt_size = 1
    
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def train(resnet50, model1, model2, train_loader, test_loader, device, epoch):
    optimizer = optim.Adam(list(model1.parameters())+list(model2.parameters()), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    max_acc = 0
    max_ep = -1
    for ep in range(epoch):
        loss_total = 0
        train_acc = 0
        data_count = 0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            #train model_D
            #init
            data_count += imgs[0].size(0)
            labels_slice = 0
            imgs_slice = 0
            for i in range(0, labels[0].size(0), 500):
                if i + 500 <= labels[0].size(0):
                    labels_slice = labels[0][i:i+500].to(device)
                    imgs_slice = imgs[0][i:i+500].to(device)
                else:
                    labels_slice = labels[0][i:].to(device)
                    imgs_slice = imgs[0][i:].to(device)
                model1.train()
                model2.train()
                resnet50.eval()
                optimizer.zero_grad()
                
                #get predict labels
                features = resnet50(imgs_slice)
                #np.save("task3F_eval/Train-Features-%d" % batch_idx, features.cpu().numpy())
                
                #features = np.load("task1F_eval/Train-Features-%d.npy" % batch_idx)
                #features = torch.from_numpy(features).to(device)
                rnn_hidddens = model1(features)
                pre_labels = model2(rnn_hidddens)
                        
                #loss
                label_loss = criterion(pre_labels, labels_slice)
                        
                #postit
                label_loss.backward()
                optimizer.step()
                loss_total += label_loss.item()
                        
                #acc
                pre_labels = pre_labels.data.max(1, keepdim=True)[1]
                train_acc += pre_labels.eq(labels_slice.data.view_as(pre_labels)).cpu().sum()
                
                    
            if batch_idx % 5 == 0:
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), label_loss.item() / bt_size))
                 
        if ep % 1 == 0:           
            valid_loss, acc = task3Test.test(resnet50, model1, model2, test_loader, device)
            print('Train Accuracy: {}/{}\t\t\t({:.2f}%)'.format(train_acc, data_count, 100. * train_acc.item() / data_count))
            save_checkpoint("task3M/Model1-%d.pth" % ep, model1, optimizer)
            save_checkpoint("task3M/Model2-%d.pth" % ep, model2, optimizer)
            if acc >= max_acc:
                max_acc = acc  
                max_ep = ep
    print(max_acc, max_ep)
                
def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    #resnet50 = torchvision.models.resnet50(pretrained=True).to(device)
    #for p in resnet50.parameters():
    #    p.requires_grad = False
    resnet50 = task3Model.RESNET50().to(device)
    model1 = task3Model.RNN().to(device)
    model2 = task3Model.FC().to(device)
    
    #loader
    print('doading datas')
    trainset = task3Loader.TRIMMED('hw4_data/FullLengthVideos/videos/train', 'hw4_data/FullLengthVideos/labels/train', transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))]))
    testset = task3Loader.TRIMMED('hw4_data/FullLengthVideos/videos/valid', 'hw4_data/FullLengthVideos/labels/valid', transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))]))
    train_loader = DataLoader(trainset, batch_size=bt_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(testset, batch_size=bt_size, shuffle=False, num_workers=1)
    
    #train
    print('start training')
    train(resnet50, model1, model2, train_loader, test_loader, device, 100)

if __name__ == '__main__':
    main()
