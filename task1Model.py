import torch 
import torchvision
import torch.nn as nn

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
            
        self.fcblock = nn.Sequential(
            nn.Linear(2000, 11),
            nn.Softmax(dim=0)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.cat((x[0], x[-1]), 0)
        x = self.fcblock(x)
        return x
        
def test():
    resnet50 = torchvision.models.resnet50(pretrained=True)
    for p in resnet50.parameters():
        p.requires_grad = False
    model = FC()
    model.eval()
    x = torch.rand(3,3,224,224)
    #print(x.shape)
    x = resnet50(x)
    output = model(x)
    #print(output.shape)
    #print(output.sum())
    print(model)

if __name__ == '__main__':
    test()