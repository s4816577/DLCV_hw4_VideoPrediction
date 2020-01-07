import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv

bt_size = 1

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

def test(resnet50, model1, model2, test_loader, device):
    correct = 0
    total_loss = 0
    data_count = 0
    criterion = nn.CrossEntropyLoss()
    #resnet50 = torchvision.models.resnet50(pretrained=True).to(device)
    #for p in resnet50.parameters():
    #    p.requires_grad = False
    with torch.no_grad():
        model1.eval()
        model2.eval()
        resnet50.eval()
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            #init
            with open('task3_gt_%d.csv'%batch_idx, 'w', newline='') as csvfile1:
                with open('task3_pre_%d.csv'%batch_idx, 'w', newline='') as csvfile2:
                    writer_gt = csv.writer(csvfile1)
                    writer_pre = csv.writer(csvfile2)
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
                        features = resnet50(imgs_slice)
                        #np.save("task3F_eval/Test-Features-%d" % batch_idx, features.cpu().numpy())
                        #get predict labels
                        #features = np.load("task1F_eval/Test-Features-%d.npy" % batch_idx)
                        #features = torch.from_numpy(features).to(device)
                        rnn_hiddens = model1(features)
                        pre_labels = model2(rnn_hiddens)
                        loss = criterion(pre_labels, labels_slice)
                        total_loss += loss.item()
                                    
                        #caculate correct
                        pre_labels = pre_labels.data.max(1, keepdim=True)[1]
                        correct += pre_labels.eq(labels_slice.data.view_as(pre_labels)).cpu().sum()
                        for i in labels_slice:
                            writer_gt.writerow([i.data.cpu().numpy()])
                        for i in pre_labels:
                            writer_pre.writerow([i[0].data.cpu().numpy()])   
                        
    print('Valid Accuracy: {}/{}\t\t\t({:.2f}%)'.format(
                    correct, data_count, 100. * correct.item() / data_count))
    return total_loss / len(test_loader.dataset), 100. * correct.item() / data_count
        
def main():
    '''
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model_F = dannModels.Feature_Extractor().to(device)
    model_L = dannModels.Label_Classifier().to(device)
    model_D = dannModels.Domain_Classifier().to(device)
    load_checkpoint('mix/F-23.pth', model_F, 111)
    load_checkpoint('mix/L-23.pth', model_L, 111)
    load_checkpoint('mix/D-23.pth', model_D, 111)
    
    #usps -> mnistm -> svhn
    #dataset init and loader warper
    target_set = dannLoader.DIGIT('hw3_data/digits/mnistm/test', 'hw3_data/digits/mnistm/test.csv', transforms.ToTensor())
    target_loader = DataLoader(target_set, batch_size=bt_size, shuffle=True, num_workers=1)
    
    #Debug(model, testset_loader)
    test(model_F, model_L, target_loader, device, False)
    '''

if __name__ == '__main__':
    main()