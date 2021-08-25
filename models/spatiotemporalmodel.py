from torch import nn
from torch import nn
import breizhcrops as bzh

from torchvision import models
import torch
from torch import nn

SUPPORTED_TEMPORAL_MODELS = ["inceptiontime", "lstm", "msresnet", "starrnn", "tempcnn", "transformermodel"]
SUPPORTED_SPATIAL_MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101','resnext50_32x4d','resnext50_32x4d',
                            'wide_resnet50_2', 'mobilenet_v3_large',
                            "mobilenet_v3_small", 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
                            'vgg16_bn', 'vgg19_bn', 'vgg19', "alexnet", 'squeezenet1_0']

class SpatiotemporalModel(nn.Module):
    def __init__(self, spatial_backbone="mobilenet_v3_small", temporal_backbone="LSTM", input_dim=4,
                 num_classes=9, sequencelength=365, pretrained_spatial=True, device="cpu"):
        """
        A wrapper around torchvision (spatial) and breizhcrops models (temporal)
        """
        super(SpatiotemporalModel, self).__init__()

        if spatial_backbone != "none":
            self.spatial_encoder = SpatialEncoder(backbone=spatial_backbone , input_dim=input_dim, pretrained=pretrained_spatial)
            output_dim = self.spatial_encoder.output_dim
        else:
            output_dim = input_dim
        self.temporal_encoder = TemporalEncoder(backbone=temporal_backbone, input_dim=output_dim,
                                                num_classes=num_classes, sequencelength=sequencelength, device=device)

        self.modelname = f"{spatial_backbone}_{temporal_backbone}"
        self.to(device)

    def forward(self, x):
        if hasattr(self, "spatial_encoder"):
            x = self.spatial_encoder(x)
        x = self.temporal_encoder(x)
        return x

class SpatialEncoder(torch.nn.Module):
    def __init__(self, backbone, input_dim=4, pretrained=False):
        super(SpatialEncoder, self).__init__()
        """
        A wrapper around torchvision models with some minor modifications for >3 input dimensions and features
        """
        assert backbone in SUPPORTED_SPATIAL_MODELS, f"spatial backbone model must be a supported torchvision model {SUPPORTED_SPATIAL_MODELS}"
        if "resnet" in backbone or "resnext" in backbone:
            self.model = models.__dict__[backbone](pretrained=pretrained)

            self.output_dim = self.model.fc.in_features

            # replace first conv layer to accomodate more spectral bands
            self.model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            # remove last layer to get features instead of class scores
            modules = list(self.model.children())[:-1]
            self.model = nn.Sequential(*modules)

        elif "mobilenet_v3" in backbone:
            cnn = models.__dict__[backbone](pretrained=pretrained).features
            cnn[0][0] = nn.Conv2d(input_dim, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model = nn.Sequential(
                cnn,
                nn.AdaptiveAvgPool2d((1,1))
            )
            self.output_dim = cnn[-1][0].out_channels
        elif "vgg" in backbone:
            self.model = models.__dict__[backbone](pretrained=pretrained)
            self.model.features[0] = nn.Conv2d(input_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.output_dim = self.model.classifier[-1].out_features
        elif "alexnet" in backbone:
            self.model = models.__dict__[backbone](pretrained=pretrained)
            self.model.features[0] = nn.Conv2d(input_dim, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            self.output_dim = self.model.classifier[-1].out_features
        elif "squeezenet" in backbone:
            self.model = models.__dict__[backbone](pretrained=pretrained)
            self.model.features[0] = nn.Conv2d(input_dim, 96, kernel_size=(7, 7), stride=(2, 2))
            self.output_dim = self.model.classifier[1].out_channels

        self.modelname = backbone.replace('_','-')

    def forward(self, x):
        N, T, D, H, W = x.shape
        x = self.model(x.view(N * T, D, H, W))
        return x.view(N, T, x.shape[1])

class TemporalEncoder(nn.Module):
    def __init__(self, backbone, input_dim, num_classes, sequencelength, device):
        super(TemporalEncoder, self).__init__()
        """
        A wrapper around Breizhcrops models for time series classification
        """
        backbone = backbone.lower() # make case insensitive
        assert backbone in SUPPORTED_TEMPORAL_MODELS, f"temporal backbone model must be a supported breizhcrops model {SUPPORTED_TEMPORAL_MODELS}"

        if backbone == "lstm":
            self.model = bzh.models.LSTM(input_dim=input_dim, num_classes=num_classes)
        if backbone == "inceptiontime":
            self.model = bzh.models.InceptionTime(input_dim=input_dim, num_classes=num_classes, device=device)
        if backbone == "msresnet":
            self.model = bzh.models.MSResNet(input_dim=input_dim, num_classes=num_classes)
        if backbone == "starrnn":
            self.model = bzh.models.StarRNN(input_dim=input_dim, num_classes=num_classes, device=device)
        if backbone == "tempcnn":
            self.model = bzh.models.TempCNN(input_dim=input_dim, num_classes=num_classes, sequencelength=sequencelength)
        if backbone == "transformermodel":
            self.model = bzh.models.TransformerModel(input_dim=input_dim, num_classes=num_classes)

        self.modelname = backbone

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = SpatiotemporalModel()

    X = torch.ones([12, 365, 4, 32, 32])
    y_pred = model(X)
    print(y_pred)
