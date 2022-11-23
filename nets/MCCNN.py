import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

# 搭建神经网络

# 卷积-标准化-激活-最大池化
class conv1d_bn_relu_maxpool(nn.Module):
    def __init__(self, ch_in, ch_out, k, p, s):
        super(conv1d_bn_relu_maxpool, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.k = k #卷积核大小
        self.p = p #padding
        self.s = s #stride 步幅

        self.conv1=nn.Conv1d(in_channels=ch_in,out_channels=ch_out,kernel_size=k,stride=s,padding=p)
        self.bn1 = nn.BatchNorm1d(num_features=ch_out)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2,padding=0)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        return x


class WCCNN(nn.Module):
    def __init__(self):
        super(WCCNN, self).__init__()

        self.conv01 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=100,stride=1,padding=0)
        self.conv02 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=200, stride=1, padding=0)
        self.conv03 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=300, stride=1, padding=0)
        self.conv_bn_relu_maxpool1=conv1d_bn_relu_maxpool(ch_in=1,ch_out=8,k=8,p=1,s=2)
        self.conv_bn_relu_maxpool2=conv1d_bn_relu_maxpool(ch_in=8,ch_out=8,k=32,p=1,s=4)
        self.conv_bn_relu_maxpool3=conv1d_bn_relu_maxpool(ch_in=8,ch_out=8,k=16,p=1,s=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=8 * 14, out_features=13, bias=True)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x=torch.cat((self.conv01(x),self.conv02(x),self.conv03(x)),-1)
        x=self.conv_bn_relu_maxpool1(x)
        x=self.conv_bn_relu_maxpool2(x)
        x=self.conv_bn_relu_maxpool3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model=WCCNN() #实例化网络模型
    wang_DS_RGB=Model.to(device) #将模型转移到cuda上
    input=torch.ones((64,1,985)) #生成一个batchsize为64的，通道数为1，宽度为2048的信号
    input=input.to(device) #将数据转移到cuda上
    output=Model(input) #将输入喂入网络中进行处理
    print(output.shape)
    summary(Model,input_size=(1,985)) #输入一个通道为1的宽度为2048，并展示出网络模型结构和参数
