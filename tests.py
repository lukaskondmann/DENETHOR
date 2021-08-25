import pytest
from itertools import product

import torch
from .models import SpatiotemporalModel, SUPPORTED_TEMPORAL_MODELS, SUPPORTED_SPATIAL_MODELS

TEST_SPATIAL_MODELS = ['resnet18', 'vgg11', "alexnet", 'squeezenet1_0']


@pytest.mark.parametrize("spatial_backbone,temporal_backbone",
                         product(TEST_SPATIAL_MODELS,
                                 SUPPORTED_TEMPORAL_MODELS,
                            )
                         )
def test_spatiotemporalmodel(spatial_backbone, temporal_backbone):
    ndims = 4
    num_classes = 9
    sequencelength = 365
    batchsize = 4
    imagesize = 64

    # test __init__
    model = SpatiotemporalModel(spatial_backbone, temporal_backbone, input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength)

    # test forward
    y_pred = model(torch.ones(batchsize, sequencelength, ndims, imagesize, imagesize))

    # test number of correct classes
    assert y_pred.shape[-1] == num_classes

    # test if model return logprobabilities
    assert y_pred.exp().sum(1).detach().cpu().numpy() == pytest.approx(1)

    del model
