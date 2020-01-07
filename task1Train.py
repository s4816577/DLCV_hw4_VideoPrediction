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
import task1Test
import csv

bt_size = 1
    
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def train(resnet50, model, train_loader, test_loader, device, epoch):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  
    criterion = nn.CrossEntropyLoss()
    max_acc = 0
    max_ep = -1
    with open('task1_loss.csv', 'w', newline='') as csvfile1:
        with open('task1_acc.csv', 'w', newline='') as csvfile2:
            writer_loss = csv.writer(csvfile1)
            writer_loss.writerow(['epoch', 'Train_Loss', 'Valid_Loss'])
            writer_acc = csv.writer(csvfile2)
            writer_acc.writerow(['epoch', 'Train_Accuracy', 'Valid_Accuracy'])
            for ep in range(epoch):
                loss_total = 0
                train_acc = 0
                for batch_idx, (imgs, labels) in enumerate(train_loader):
                    #train model_D
                    #init
                    labels = labels.to(device)
                    imgs = imgs[0].to(device)
                    model.train()
                    optimizer.zero_grad()
                    
                    #get predict labels
                    #features = resnet50(imgs)
                    #np.save("task1F/Train-Features-%d" % batch_idx, features.cpu().numpy())
                    features = np.load("task1F_eval/Train-Features-%d.npy" % batch_idx)
                    features = torch.from_numpy(features).to(device)
                    pre_labels = model(features)
                    pre_labels = pre_labels.view(1, -1)
                    
                    #loss
                    label_loss = criterion(pre_labels, labels)
                    loss_total += label_loss.item()
                    
                    #acc
                    pre_labels = pre_labels.data.max(1, keepdim=True)[1]
                    train_acc += pre_labels.eq(labels.data.view_as(pre_labels)).cpu().sum()
                    
                    #postit
                    label_loss.backward()
                    optimizer.step()
                    
                    if batch_idx % 500 == 0:
                        print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                            ep, batch_idx, len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), label_loss.item() / bt_size))
                 
                if ep % 1 == 0:           
                    valid_loss, acc = task1Test.test(model, test_loader, device)
                    #save_checkpoint("task1M/Model-%d.pth" % ep, model, optimizer)
                    if acc >= max_acc:
                        max_acc = acc
                        max_ep = ep
                    writer_acc.writerow([ep, 100. * train_acc.item() / len(train_loader.dataset), acc])
                    writer_loss.writerow([ep, loss_total / len(train_loader.dataset), valid_loss])
                    print('Train_acc:{:.2f}%'.format(100. * train_acc.item() / len(train_loader.dataset)))
            print(max_acc, max_ep)
            
def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model = task1Model.FC().to(device)
    resnet50 = torchvision.models.resnet50(pretrained=True).to(device)
    for p in resnet50.parameters():
        p.requires_grad = False
    
    #loader
    print('doading datas')
    trainset = task1Loader.TRIMMED('hw4_data/TrimmedVideos/video/train', 'hw4_data/TrimmedVideos/label/gt_train.csv', transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))]))
    testset = task1Loader.TRIMMED('hw4_data/TrimmedVideos/video/valid', 'hw4_data/TrimmedVideos/label/gt_valid.csv', transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))]))
    train_loader = DataLoader(trainset, batch_size=bt_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(testset, batch_size=bt_size, shuffle=False, num_workers=1)
    
    #train
    print('start training')
    train(resnet50, model, train_loader, test_loader, device, 200)

if __name__ == '__main__':
    main()
