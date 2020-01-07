import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import task3Model
import task3Loader
import task3Test
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
            current_correct = 0
            with open('task3_gt_%d.csv'%batch_idx, 'w', newline='') as csvfile1:
                with open('task3_pre_%d.csv'%batch_idx, 'w', newline='') as csvfile2:
                    writer_gt = csv.writer(csvfile1)
                    writer_pre = csv.writer(csvfile2)
                    data_count += imgs[0].size(0)
                    labels_slice = 0
                    imgs_slice = 0
                    for i in range(0, labels[0].size(0), 300):
                        if i + 300 <= labels[0].size(0):
                            labels_slice = labels[0][i:i+300].to(device)
                            imgs_slice = imgs[0][i:i+300].to(device)
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
                        current_correct += pre_labels.eq(labels_slice.data.view_as(pre_labels)).cpu().sum()
                        for i in labels_slice:
                            writer_gt.writerow([i.data.cpu().numpy()])
                        for i in pre_labels:
                            writer_pre.writerow([i[0].data.cpu().numpy()])   
            print('Current_Acc: {}/{}  {:.2f}%'.format( current_correct, imgs[0].size(0), 100. * current_correct / imgs[0].size(0)))
    print('Valid Accuracy: {}/{}\t\t\t({:.2f}%)'.format(
                    correct, data_count, 100. * correct.item() / data_count))
    return total_loss / len(test_loader.dataset), 100. * correct.item() / data_count
        
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
    load_checkpoint('task3M/Model1-83.pth', model1, 111)
    load_checkpoint('task3M/Model2-83.pth', model2, 111)
    
    #loader
    print('doading datas')
    testset = task3Loader.TRIMMED('hw4_data/FullLengthVideos/videos/valid', 'hw4_data/FullLengthVideos/labels/valid', transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))]))
    test_loader = DataLoader(testset, batch_size=bt_size, shuffle=False, num_workers=1)
    
    #test
    print('start training')
    test(resnet50, model1, model2, test_loader, device)
    

if __name__ == '__main__':
    main()
