import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

vgg_cfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class vgg(nn.Module):
    def __init__(self, num_class, depth=16, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = vgg_cfg[depth]

        self.cfg = cfg
        self.feature = self.make_layers(self.cfg, True)

        self.classifier = nn.Sequential(
              nn.Linear(cfg[-2], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              nn.Linear(512, num_class)
            )
        if init_weights:
            self.initialize_weights()
    
    def conv3x3_dw(self, input_filters, output_filters):
        return nn.Sequential(
               nn.Conv2d(input_filters, input_filters, 3, 1, 1, groups=input_filters, bias=False),
               nn.BatchNorm2d(input_filters),
               nn.ReLU6(inplace=True),
                
               nn.Conv2d(input_filters, output_filters, 1, 1, 0, bias=False),
               nn.BatchNorm2d(output_filters),
               nn.ReLU6(inplace=True),
        )

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(x.size(2))(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  

class vgg_dw(nn.Module):
    def __init__(self, num_class, depth=16, init_weights=True, cfg=None):
        super(vgg_dw, self).__init__()
        if cfg is None:
            cfg = vgg_cfg[depth]

        self.cfg = cfg
        self.feature = self.make_layers(self.cfg, True)

        self.classifier = nn.Sequential(
              nn.Linear(cfg[-2], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              nn.Linear(512, num_class)
            )
        if init_weights:
            self.initialize_weights()
    
    def conv3x3_dw(self, input_filters, output_filters):
        return nn.Sequential(
               nn.Conv2d(input_filters, input_filters, 3, 1, 1, groups=input_filters, bias=False),
               nn.BatchNorm2d(input_filters),
               nn.ReLU6(inplace=True),
                
               nn.Conv2d(input_filters, output_filters, 1, 1, 0, bias=False),
               nn.BatchNorm2d(output_filters),
               nn.ReLU6(inplace=True),
        )

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = self.conv3x3_dw(in_channels,v)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(x.size(2))(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  