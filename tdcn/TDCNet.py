import torch
import torch.nn as nn

class MRB_Block(nn.Module):
    def __init__(self):
        super(MRB_Block, self).__init__()
        self.conv_3 = nn.Sequential( nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv_5_1 = nn.Sequential( nn.Conv2d(in_channels=64, out_channels=64, kernel_size= 3, stride= 1, padding = 2, dilation =2 , bias=True))
        self.conv_5_2 = nn.Sequential( nn.Conv2d(in_channels=64, out_channels=64, kernel_size= 3, stride= 1, padding = 3, dilation =3 , bias=True))
        self.confusion = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = x
        output_3 = self.conv_3(x)
        output_5_1 = self.conv_5_1(output_3)
        output_5_2 = self.conv_5_2(output_5_1)
        output = torch.cat([output_3, output_5_1, output_5_2], 1) 
        output = self.confusion(output)
        output = self.relu(output)
        output = torch.add(output, identity_data)
        return output
        
        
class TDCNet(nn.Module):
    def __init__(self,base_filter):
        super(TDCNet, self).__init__()

        self.sample = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=base_filter, kernel_size=32, stride=32, padding=0, bias=False))

        self.initialization = torch.nn.Sequential(nn.ConvTranspose2d(in_channels=base_filter, out_channels=1, kernel_size=32, stride=32, padding=0, bias=False))

        self.getFactor = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.residual1 = self.make_layer(MRB_Block)
        self.residual2 = self.make_layer(MRB_Block)
        self.residual3 = self.make_layer(MRB_Block)
        self.residual4 = self.make_layer(MRB_Block)
        self.residual5 = self.make_layer(MRB_Block)
        self.residual6 = self.make_layer(MRB_Block) 
        self.residual7 = self.make_layer(MRB_Block) 
        self.add = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    

    def forward(self, x):
        out = self.initialization(self.sample(x))
        LR = self.getFactor(out)
        out1 = self.residual1(LR)
        out2 = self.residual2(out1)
        out3 = self.residual3(out2)
        out4 = self.residual4(out3)
        out5 = self.residual5(out4)
        out6 = self.residual6(out5)
        out7 = self.residual7(out6)   
        out = torch.cat([out1,out2,out3,out4,out5,out6,out7,LR], 1)
        out = self.relu (self.add(out))
        out = self.relu (self.conv(out)) + LR
        out = self.output(out) 

        return out

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
