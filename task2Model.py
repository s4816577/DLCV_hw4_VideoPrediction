import torch 
import torchvision
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
            
        self.rnn = nn.LSTM(
            input_size=1000,
            hidden_size=128,
            num_layers=4,
            batch_first=True,
            dropout=0.5,
            #bidirectional=True
        )
        
        self.out = nn.Sequential(
            nn.Linear(128, 11),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = x.view(1,-1,1000)
        x, (h_n, c_n) = self.rnn(x, None)
        print(x[:,-1,:] == h_n[3])
        print(h_n[3].shape)
        print(x[:,-1,:].shape)
        x = self.out(x[:,-1,:])
        return x
        
def test():
    resnet50 = torchvision.models.resnet50(pretrained=True)
    for p in resnet50.parameters():
        p.requires_grad = False
    model = RNN()
    model.eval()
    x = torch.rand(3,3,224,224)
    x = resnet50(x)
    print(x.shape)
    output = model(x)
    #print(output)
    #print(output.shape)
    #print(model)

if __name__ == '__main__':
    test()