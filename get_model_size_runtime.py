from itertools import product

import torch
from models import SpatiotemporalModel, SUPPORTED_TEMPORAL_MODELS, SUPPORTED_SPATIAL_MODELS
from time import time

ndims = 4
num_classes = 9
sequencelength = 365
batchsize = 2
imagesize = 32

import warnings
warnings.filterwarnings("ignore")

print(f"# simulated batchsize={batchsize}, imagesize={imagesize}, sequencelength={sequencelength}; time=-1: test failed (likely OOM error)")
print(f"spatial_backbone, temporal_backbone, num_params, forward_time_cpu, backward_time_cpu, forward_time_cuda, backward_time_cuda, init_time")
for spatial_backbone, temporal_backbone in product(SUPPORTED_SPATIAL_MODELS, SUPPORTED_TEMPORAL_MODELS):
    # test __init__

    start = time()
    model = SpatiotemporalModel(spatial_backbone, temporal_backbone, input_dim=ndims, num_classes=num_classes,
                                sequencelength=sequencelength, pretrained_spatial=False)
    init_time = time() - start

    # test forward
    start = time()
    y_pred = model(torch.ones(batchsize, sequencelength, ndims, imagesize, imagesize))
    forward_time_cpu = time() - start

    # test backward
    start = time()
    y_pred.mean().backward()
    backward_time_cpu = time() - start

    try:
        model = SpatiotemporalModel(spatial_backbone, temporal_backbone, input_dim=ndims, num_classes=num_classes,
                                    sequencelength=sequencelength, pretrained_spatial=False, device="cuda")
        model = model.to("cuda")
        X = torch.ones(batchsize, sequencelength, ndims, imagesize, imagesize).to("cuda")
        # test forward
        start = time()
        y_pred = model(X)
        forward_time_cuda = time() - start

        # test backward
        start = time()
        y_pred.mean().backward()
        backward_time_cuda = time() - start
    except RuntimeError:
        # likely out of memory error
        forward_time_cuda = -1
        backward_time_cuda = -1
        pass

    # number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{spatial_backbone}, {temporal_backbone}, {num_params}, {forward_time_cpu:.4f}, {backward_time_cpu:.4f}, {forward_time_cuda:.4f}, {backward_time_cuda:.4f}, {init_time:.4f}")
