import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = [
    'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, non_linearity, bias, classifier_bias, num_classes=1000, width_scaling=1,
                 init_weights=True, **non_linearity_kwargs):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(int(width_scaling * 512 * 7 * 7), int(width_scaling * 4096), bias=bias),
            non_linearity(num_channels=int(width_scaling * 4096), **non_linearity_kwargs),
            nn.Dropout(),
            nn.Linear(int(width_scaling * 4096), int(width_scaling * 4096), bias=bias),
            non_linearity(num_channels=int(width_scaling * 4096), **non_linearity_kwargs),
            nn.Dropout(),
            nn.Linear(int(width_scaling * 4096), num_classes, bias=classifier_bias),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, non_linearity, bias, batch_norm, width_scaling, **non_linearity_kwargs):
    """
    :param cfg:
    :param non_linearity:
    :param bias: whether there is learned bias (affine parameter of batch norms)
    :param batch_norm: can be "none", "pre", or "post" (non-linearity)
    :param width_scaling: optional scaling factor applied on all layer widths
    :param non_linearity_kwargs:
    :return:
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, int(width_scaling * v), kernel_size=3,
                                 padding=1, bias=bias and not batch_norm)]
            if batch_norm == "pre":
                layers += [nn.BatchNorm2d(int(width_scaling * v), affine=bias)]
            layers += [non_linearity(num_channels=int(width_scaling * v), **non_linearity_kwargs)]
            if batch_norm == "post":
                layers += [nn.BatchNorm2d(int(width_scaling * v), affine=bias)]
            in_channels = int(width_scaling * v)
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(config, batch_norm, non_linearity, bias, classifier_bias, width_scaling,
        init_weights=True, pretrained=False, **non_linearity_kwargs):
    """ vgg constructor, to factorize other models.
    :param config: list of layers
    :param batch_norm: can be False, "pre", or "post" (non-linearity)
    :param non_linearity: non-linearity constructor
    :param bias: whether to include bias (in convs and affine batch norms)
    :param classifier_bias! whether to include bias in last classifier layer
    """
    if pretrained:
        init_weights = False
    model = VGG(features=make_layers(cfg=config, non_linearity=non_linearity, bias=bias, batch_norm=batch_norm,
                                     width_scaling=width_scaling, **non_linearity_kwargs),
                non_linearity=non_linearity, bias=bias, classifier_bias=classifier_bias, width_scaling=width_scaling,
                init_weights=init_weights, **non_linearity_kwargs)
    if pretrained:
        assert batch_norm != "post"  # "pre" or "none"
        assert bias and classifier_bias
        assert width_scaling == 1
        url = f"vgg{len(config) - 2}{'_bn' if batch_norm == 'pre' else ''}"
        model.load_state_dict(model_zoo.load_url(model_urls[url]))
    return model


def vgg11(non_linearity, batch_norm, bias, classifier_bias, width_scaling,
          init_weights=True, pretrained=False, **non_linearity_kwargs):
    return vgg(config=cfg['A'], batch_norm=batch_norm, non_linearity=non_linearity, bias=bias,
               classifier_bias=classifier_bias, width_scaling=width_scaling,
               init_weights=init_weights, pretrained=pretrained, **non_linearity_kwargs)


def vgg13(non_linearity, batch_norm, bias, classifier_bias, width_scaling,
          init_weights=True, pretrained=False, **non_linearity_kwargs):
    return vgg(config=cfg['B'], batch_norm=batch_norm, non_linearity=non_linearity, bias=bias,
               classifier_bias=classifier_bias, width_scaling=width_scaling,
               init_weights=init_weights, pretrained=pretrained, **non_linearity_kwargs)


def vgg16(non_linearity, batch_norm, bias, classifier_bias, width_scaling,
          init_weights=True, pretrained=False, **non_linearity_kwargs):
    return vgg(config=cfg['D'], batch_norm=batch_norm, non_linearity=non_linearity, bias=bias,
               classifier_bias=classifier_bias, width_scaling=width_scaling,
               init_weights=init_weights, pretrained=pretrained, **non_linearity_kwargs)


def vgg19(non_linearity, batch_norm, bias, classifier_bias, width_scaling,
          init_weights=True, pretrained=False, **non_linearity_kwargs):
    return vgg(config=cfg['E'], batch_norm=batch_norm, non_linearity=non_linearity, bias=bias,
               classifier_bias=classifier_bias, width_scaling=width_scaling,
               init_weights=init_weights, pretrained=pretrained, **non_linearity_kwargs)
