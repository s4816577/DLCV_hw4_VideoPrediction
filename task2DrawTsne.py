import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import task2Loader
import task2Model

COLOR_POOL = ('black', 'orange', 'purple', 'red', 'sienna', 'green', 'blue', 'grey', 'pink', 'yellow', 'cyan') 
bt_size = 1

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path, map_location='cuda:1')
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
        
def plot_embedding(X, y):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig1 = plt.figure(figsize=(10, 10))
    for i in range(len(y)): 
        colors = COLOR_POOL[y[i]]
        plt.scatter(X[i, 0], X[i, 1], color=colors)
    fig_name = 'task2_tsne.png'
    plt.savefig(fig_name)
    print('{} is saved'.format(fig_name))

def Plot(resnet50, model, test_loader, device):
    feature_list = []
    label_list = []
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    model.eval()
    resnet50.eval()
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            #get features on valid
            labels = labels[0].to(device)
            features = np.load("task1F_eval/Test-Features-%d.npy" % batch_idx)
            features = torch.from_numpy(features).to(device)
            merged_features = model(features)
            
            #add list
            feature_list.append(merged_features[0])
            label_list.append(labels)
    
    label_list = torch.stack(label_list)
    feature_list = torch.stack(feature_list)
    feature_list = tsne.fit_transform(feature_list.detach().cpu().numpy())
    plot_embedding(feature_list, label_list.detach().cpu().numpy())
            
def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model = task2Model.RNN().to(device)
    load_checkpoint('task2M/Model-438.pth', model, 111)
    resnet50 = torchvision.models.resnet50(pretrained=True).to(device)
    for p in resnet50.parameters():
        p.requires_grad = False
    
    #loader
    print('doading datas')
    testset = task2Loader.TRIMMED('hw4_data/TrimmedVideos/video/valid', 'hw4_data/TrimmedVideos/label/gt_valid.csv', transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229,0.224,0.225))]))
    test_loader = DataLoader(testset, batch_size=bt_size, shuffle=False, num_workers=1)
    
    #train
    print('start training')
    Plot(resnet50, model, test_loader, device)

if __name__ == '__main__':
    main()
