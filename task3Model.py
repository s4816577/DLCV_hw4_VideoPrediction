import torch 
import torchvision
import torch.nn as nn

class RESNET50(nn.Module):
    def __init__(self):
        super(RESNET50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        module = list(resnet50.children())[:-1]
        self.Resnet50 = nn.Sequential(*module)
        for p in self.Resnet50.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        return self.Resnet50(x).view(x.size(0), -1)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
            
        self.rnn = nn.LSTM(
            input_size=2048,
            hidden_size=128,
            num_layers=4,
            batch_first=True,
            dropout=0.5,
            #bidirectional=True
        )
        
    def forward(self, x):
        x = x.view(1,-1,2048)
        x, _ = self.rnn(x, None)
        return x.view(-1, 128)
    
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(128, 11)
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x
        
def test():
    resnet50 = RESNET50()
    model = RNN()
    model.eval()
    resnet50.eval()
    x = torch.rand(3,3,224,224)
    x = resnet50(x)
    #print(x.shape)
    x = model(x)
    print(model)
    #print(x.shape)
    model = FC()
    x = model(x)
    print(model)
    #print(x)
    #print(x.sum(1))

if __name__ == '__main__':
    test()
