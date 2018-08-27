import torch.nn as nn
import numpy as np 
import torch

class Encoder(nn.Module):
    def __init__(self ):
        super(Encoder,self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,16,3,padding=1, stride = 1),   # batch x 16 x 28 x 28
                        nn.ReLU(),
                        nn.Conv2d(16,64,3,padding=1 , stride = 1),  # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.Conv2d(64,64,3,padding=1 , stride = 1),  # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.MaxPool2d(2,2)   # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1 , stride = 1),  # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128,128,3,padding=1 , stride = 1),  # batch x 64 x 7 x 7
                        nn.ReLU(),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128,256,3,padding=1 , stride = 1),  # batch x 64 x 7 x 7
                        nn.ReLU()

        )
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        return  out


class Decoder(nn.Module):
    def __init__(self ):
        super(Decoder,self).__init__()

        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,1,1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(128,128,3,2,1,1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(128,64,3,1,1),
                        nn.ReLU()
            )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(64,64,3,1,1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64,16,3,1,1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(16,3,3,2,1,1),
                        nn.ReLU()
        )
        
    def forward(self,x):
        out = x.view(x.shape[0] , 256, np.sqrt(x.shape[1] / 256) , np.sqrt(x.shape[1] / 256))
        out = self.layer1(out)
        out = self.layer2(out)
        return out