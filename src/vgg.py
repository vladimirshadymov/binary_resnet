from bnn_modules import Binarization, BinarizedConv2d, BinarizedLinear
import torch
import torch.nn as nn

cfgs = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def fc_layer(in_features, out_features, binarized=False):
    if binarized:
        return BinarizedLinear(in_features, out_features)
    else:
        return nn.Linear(in_features, out_features)

def activation(binarized=False):
    if binarized:
        return Binarization()
    else:
        nn.ReLU(True)

def get_conv2d(in_channels, out_channels, kernel_size=3, padding=1, binarized=False):
    if binarized:
        return BinarizedConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)



class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, binarized=False):
        super(VGG, self).__init__()
        self.binarized = binarized
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            fc_layer(512 * 7 * 7, 4096, binarized=self.binarized),
            activation(binarized=self.binarized),
            nn.Dropout(),
            fc_layer(4096, 4096, binarized=self.binarized),
            activation(binarized=self.binarized),
            nn.Dropout(),
            fc_layer(4096, num_classes, binarized=self.binarized),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  #TODO: check the influence of non-linearity
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, BinarizedConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  #TODO: check the influence of non-linearity
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BinarizedLinear):
                nn.init.normal_(m.weight, 0, 0.01) #TODO: check if code works at this IF or it percieves
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(cfg, binarized=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = get_conv2d(in_channels, v, kernel_size=3, padding=1, binarized=binarized)
            layers += [conv2d, nn.BatchNorm2d(v), activation(binarized=binarized)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(arch, cfg, **kwargs):
    model = VGG(make_layers(cfgs[cfg]), **kwargs)
    return model

def vgg16(**kwargs):
    r"""VGG 16-layer model 
    Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>
    """
    return _vgg('vgg16', 'A', **kwargs)


def vgg19(**kwargs):
    r"""VGG 19-layer model. 
    Very Deep Convolutional Networks For Large-Scale Image Recognition <https://arxiv.org/pdf/1409.1556.pdf>
    """
    return _vgg('vgg19', 'B', **kwargs)
