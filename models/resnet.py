import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

scale_first_layer = True


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, non_linearity, bias, batch_norm,
                 stride=1, downsample=None, **non_linearity_kwargs):
        """
        :param inplanes:
        :param planes:
        :param non_linearity:
        :param bias: whether to include bias in convs (and affine parameter of batch_norm)
        :param batch_norm: can be "none", "pre", or "post"
        :param stride:
        :param downsample:
        :param non_linearity_kwargs:
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        if batch_norm != "none":
            self.bn1 = nn.BatchNorm2d(planes, affine=bias)
        self.non_linearity1 = non_linearity(num_channels=planes, **non_linearity_kwargs)
        self.conv2 = conv3x3(planes, planes)
        if batch_norm != "none":
            self.bn2 = nn.BatchNorm2d(planes, affine=bias)
        self.non_linearity2 = non_linearity(num_channels=planes, **non_linearity_kwargs)
        self.batch_norm = batch_norm
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.batch_norm == "pre":
            out = self.bn1(out)
        out = self.non_linearity1(out)
        if self.batch_norm == "post":
            out = self.bn1(out)

        out = self.conv2(out)
        if self.batch_norm == "pre":
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.non_linearity2(out)
        if self.batch_norm == "post":
            out = self.bn2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, non_linearity, bias, batch_norm,
                 stride=1, downsample=None, **non_linearity_kwargs):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        if batch_norm != "none":
            self.bn1 = nn.BatchNorm2d(planes, affine=bias)
        self.non_linearity1 = non_linearity(num_channels=planes, **non_linearity_kwargs)
        self.conv2 = conv3x3(planes, planes, stride)
        if batch_norm != "none":
            self.bn2 = nn.BatchNorm2d(planes, affine=bias)
        self.non_linearity2 = non_linearity(num_channels=planes, **non_linearity_kwargs)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        if batch_norm != "none":
            self.bn3 = nn.BatchNorm2d(planes * self.expansion, affine=bias)
        self.non_linearity3 = non_linearity(num_channels=planes, **non_linearity_kwargs)
        self.downsample = downsample
        self.stride = stride
        self.batch_norm = batch_norm

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.batch_norm == "pre":
            out = self.bn1(out)
        out = self.non_linearity1(out)
        if self.batch_norm == "post":
            out = self.bn1(out)

        out = self.conv2(out)
        if self.batch_norm == "pre":
            out = self.bn2(out)
        out = self.non_linearity2(out)
        if self.batch_norm == "post":
            out = self.bn2(out)

        out = self.conv3(out)
        if self.batch_norm == "pre":
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.non_linearity3(out)
        if self.batch_norm == "post":
            out = self.bn3(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, non_linearity, batch_norm, bias, classifier_bias, width_scaling=1,
                 num_classes=1000, zero_init_residual=False, **non_linearity_kwargs):
        super(ResNet, self).__init__()
        
        w_first_layer = width_scaling if scale_first_layer else 1
        self.inplanes = int(w_first_layer * 64)
        self.conv1 = nn.Conv2d(3, int(w_first_layer * 64), kernel_size=7, stride=2, padding=3,
                               bias=False)
        if batch_norm != "none":
            self.bn1 = nn.BatchNorm2d(int(w_first_layer * 64), affine=bias)
        self.non_linearity = non_linearity(num_channels=int(w_first_layer * 64), **non_linearity_kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(width_scaling * 64), layers[0], non_linearity, batch_norm, bias, **non_linearity_kwargs)
        self.layer2 = self._make_layer(block, int(width_scaling * 128), layers[1], non_linearity, batch_norm, bias, stride=2, **non_linearity_kwargs)
        self.layer3 = self._make_layer(block, int(width_scaling * 256), layers[2], non_linearity, batch_norm, bias, stride=2, **non_linearity_kwargs)
        self.layer4 = self._make_layer(block, int(width_scaling * 512), layers[3], non_linearity, batch_norm, bias, stride=2, **non_linearity_kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(width_scaling * 512 * block.expansion), num_classes, bias=classifier_bias)
        self.batch_norm = batch_norm

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and bias:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual and bias:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, non_linearity, batch_norm, bias, stride=1, **non_linearity_kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [conv1x1(self.inplanes, planes * block.expansion, stride)]
            if batch_norm == "pre":  # No batch_norm in post, taken care of in BasicBlock/Bottleneck
                layers += [nn.BatchNorm2d(planes * block.expansion, affine=bias)]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, non_linearity=non_linearity, bias=bias,
                            batch_norm=batch_norm, stride=stride, downsample=downsample, **non_linearity_kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, non_linearity=non_linearity, bias=bias,
                                batch_norm=batch_norm, **non_linearity_kwargs))  # stride=1, downsample=None

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm == "pre":
            x = self.bn1(x)
        x = self.non_linearity(x)
        if self.batch_norm == "post":
            x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(name, block, sizes, non_linearity, batch_norm, bias, classifier_bias, width_scaling,
           pretrained=False, **non_linearity_kwargs):
    """ Factorizes other ResNet calls. """
    model = ResNet(block=block, layers=sizes, non_linearity=non_linearity, batch_norm=batch_norm,
                   bias=bias, classifier_bias=classifier_bias, width_scaling=width_scaling, **non_linearity_kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[name]))
    return model


def resnet18(pretrained=False, **kwargs):
    return resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained=pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
    return resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained=pretrained, **kwargs)


def resnet50(pretrained=False, **kwargs):
    return resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained=pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    return resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained=pretrained, **kwargs)


def resnet152(pretrained=False, **kwargs):
    return resnet("resnet101", Bottleneck, [3, 8, 63, 3], pretrained=pretrained, **kwargs)
