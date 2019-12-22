import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1 # 是对输出深度的倍乘，在这里等同于忽略

    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # 批标准化层
        self.relu = nn.ReLU(True) # 激活函数

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample # 这个是shortcut的操作
        self.stride = stride # 得到步长

    def forward(self,x):
        residual = x # 获得上一层的输出

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None: # 当shortcut存在的时候
            residual = self.downsample(x) 
            # 我们将上一层的输出x输入进这个downsample所拥有一些操作（卷积等），将结果赋给residual
            # 简单说，这个目的就是为了应对上下层输出输入深度不一致问题

        out += residual # 将bn2的输出和shortcut过来加在一起
        out = self.relu(out)

        return out

class Bottleneck(nn.Module): # 由于bottleneck译意为瓶颈，我这里就称它为瓶颈块
    expansion = 4 # 若我们输入深度为64，那么扩张4倍后就变为了256
    # 其目的在于使得当前块的输出深度与下一个块的输入深度保持一致
    # 而为什么是4，这是因为在设计网络的时候就规定了的
    # 我想应该可以在保证各层之间的输入输出一致的情况下修改扩张的倍数
    
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        # 这层1*1卷积层，是为了降维，把输出深度降到与3*3卷积层的输入深度一致
        
        self.conv2 = nn.conv3x3(out_channels,out_channels) # 3*3卷积操作
        self.bn2 = nn. BatchNorm2d(out_channels)
        # 这层3*3卷积层的channels是下面_make_layer中的第二个参数规定的
        
        self.conv3 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        # 这层1*1卷积层，是在升维，四倍的升

        self.relu = nn.ReLU(True) # 激活函数
        self.downsample = downsample # shortcut信号
        self.stride = stride # 获取步长

    def forward(self,x):
        residual = x

        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out) # 连接一个激活函数

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x) # 目的同上

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,block,layers,num_classes=10):
        # block:为上边的基础块BasicBlock或瓶颈块Bottleneck，它其实就是一个对象
        # layers:每个大layer中的block个数，设为blocks更好，但每一个block实际上也很是一些小layer
        # num_classes:表示最终分类的种类数
        super(ResNet,self).__init__()
        self.in_channels = 64 # 输入深度为64，我认为把这个理解为每一个残差块块输入深度最好

        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False) 
        # 输入深度为3(正好是彩色图片的3个通道)，输出深度为64，滤波器为7*7，步长为2，填充3层，特征图缩小1/2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True) # 激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # 最大池化，滤波器为3*3，步长为2，填充1层，特征图又缩小1/2
        # 此时，特征图的尺寸已成为输入的1/4

        # 下面的每一个layer都是一个大layer
        # 第二个参数是残差块中3*3卷积层的输入输出深度
        self.layer1 = self._make_layer(block,64,layers[0]) # 特征图大小不变
        self.layer2 = self._make_layer(block,128,layers[1],stride=2) # 特征图缩小1/2
        self.layer3 = self._make_layer(block,256,layers[2],stride=2) # 特征图缩小1/2
        self.layer4 = self._make_layer(block,512,layers[3],stride=2) # 特征图缩小1/2
        # 这里只设置了4个大layer是设计网络时规定的，我们也可以视情况自己往上加
        # 这里可以把4个大layer和上边的一起看成是五个阶段
        
        self.avgpool = nn.AvgPool2d(7,stride=1) # 平均池化，滤波器为7*7，步长为1，特征图大小变为1*1
        self.fc = nn.Linear(512*block.expansion,num_classes) # 全连接层

        # 这里进行的是网络的参数初始化，可以看出卷积层和批标准化层的初始化方法是不一样的
        for m in self.modules(): 
            # self.modules()采取深度优先遍历的方式，存储了网络的所有模块，包括本身和儿子
            if isinstance(m,nn.Conv2d): # isinstance()判断一个对象是否是一个已知的类型
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                # 9. kaiming_normal 初始化 (这里是nn.init初始化函数的源码，有好几种初始化方法)
                # torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
                # nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
                # tensor([[ 0.2530, -0.4382,  1.5995],
                #         [ 0.0544,  1.6392, -2.0752]])
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
                # 3. 常数 - 固定值 val
                # torch.nn.init.constant_(tensor, val)
                # nn.init.constant_(w, 0.3)
                # tensor([[ 0.3000,  0.3000,  0.3000],
                #         [ 0.3000,  0.3000,  0.3000]])

    def _make_layer(self,block,out_channels,blocks,stride=1):
        # 这里的blocks就是该大layer中的残差块数
        # out_channels表示的是这个块中3*3卷积层的输入输出深度
        downsample = None # shortcut内部的跨层实现
        if stride != 1 or self.in_channels != out_channels*block.expansion:
            # 判断步长是否为1，判断当前块的输入深度和当前块卷积层深度乘于残差块的扩张
            # 为何用步长来判断，我现在还不明白，感觉没有也行
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,out_channels*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*block.expansion) 
                )
            # 一旦判断条件成立，那么给downsample赋予一层1*1卷积层和一层批标准化层。并且这一步将伴随这特征图缩小1/2
            # 而为何要在shortcut中再进行卷积操作？是因为在残差块之间，比如当要从64深度的3*3卷积层阶段过渡到128深度的3*3卷积层阶段，主分支为64深度的输入已经通过128深度的3*3卷积层变成了128深度的输出，而shortcut分支中x的深度仍为64，而主分支和shortcut分支相加的时候，深度不一致会报错。这就需要进行升维操作，使得shortcut分支中的x从64深度升到128深度。
            # 而且需要这样操作的其实只是在基础块BasicBlock中，在瓶颈块Bottleneck中主分支中自己就存在升维操作，那么Bottleneck还在shortcut中引入卷积层的目的是什么？能带来什么帮助？


        layers = []
        layers.append(block(self.in_channels,out_channels,stride,downsample))
        # block()生成上面定义的基础块和瓶颈块的对象，并将dowsample传递给block


        self.in_channels = out_channels*block.expansion # 改变下面的残差块的输入深度
        # 这使得该阶段下面blocks-1个block，即下面循环内构造的block与下一阶段的第一个block的在输入深度上是相同的。
        for i in range(1,blocks):  # 这里面所有的block
            layers.append(block(self.in_channels,out_channels))
        #一定要注意，out_channels一直都是3*3卷积层的深度
        return nn.Sequential(*layers) # 这里表示将layers中的所有block按顺序接在一起

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out) # 写代码时一定要仔细，别把out写成x了，我在这里吃了好大的亏

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0),-1) # 将原有的多维输出拉回一维
        out = self.fc(out)

        return out