import sys

import argparse
import numpy as np

from breizhcrops.models import LSTM, TempCNN, MSResNet, TransformerModel, InceptionTime, StarRNN, OmniScaleCNN
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import torch
import pandas as pd
import os
import sklearn.metrics
from dataset import Dataset, CLASSES
from models import SpatiotemporalModel
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from transforms import random_crop, crop_or_pad_to_size
import shutil




def test(args):
    
    num_classes = 9
    ndims = 4
    sequencelength = 365

    def transform(image_stack, mask):

        if args.spatial_encoder == "none": # average over field mask: T, D = image_stack.shape
            image_stack = image_stack[:, :, mask > 0].mean(2)
            mask = -1 # mask is meaningless now but needs to be constant size for batching
        else: # crop/pad image to fixed size + augmentations: T, D, H, W = image_stack.shape
            
            if image_stack.shape[2] >= args.imagesize and image_stack.shape[3] >= args.imagesize:
                image_stack, mask = random_crop(image_stack, mask, args.imagesize)

            image_stack, mask = crop_or_pad_to_size(image_stack, mask, args.imagesize)

        image_stack = image_stack * 1e-4

        # z-normalize
        image_stack -= 0.1014 + np.random.normal(scale=0.01)
        image_stack /= 0.1171 + np.random.normal(scale=0.01)

        return torch.from_numpy(np.ascontiguousarray(image_stack)).float(), torch.from_numpy(np.ascontiguousarray(mask))

    dataset = Dataset(tifroot=args.datapath,
                      labelgeojson=args.labelgeojson,
                      transform=transform
                      )

    testdataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, num_workers=args.workers)

    device = torch.device(args.device)
    model = SpatiotemporalModel(spatial_backbone=args.spatial_encoder, temporal_backbone=args.temporal_encoder,
                                input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength, device=device)
    model.to(device)

    print(f"Initialized {model.modelname}")
    
    logdir = os.path.join(args.logdir, model.modelname)
    snapshot_path = os.path.join(logdir, "model_best.pth.tar")
    assert os.path.exists(snapshot_path), "a valid model snapshot path must be provided for evaluation"
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging results to {logdir}")

    summarywriter = SummaryWriter(log_dir=logdir)

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    log = list()
    start_epoch = 1
    # resume model if exists
    
    if os.path.exists(snapshot_path):
        checkpoint = torch.load(snapshot_path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded Best Model {snapshot_path}, which was after epoch {start_epoch}")

    test_loss, y_true, y_pred, *_ = test_epoch(model, criterion, testdataloader, device)
    test_loss = test_loss.cpu().detach().numpy()[0]
    scores = metrics(y_true.cpu(), y_pred.cpu())
    #scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
    scores["testloss"] = test_loss
    scores["model"] = snapshot_path
    log.append(scores)
    
    log_df = pd.DataFrame(log).set_index("model")
    log_df.to_csv(os.path.join(logdir, "test_scores.csv"))
    
    print("Evaluation successful. Wrote scores to test_scores.csv")

def metrics(y_true, y_pred):
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro", zero_division=0)
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")

    return dict(
        accuracy=accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
    )


def test_epoch(model, criterion, dataloader, device):
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()
        field_ids_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true, field_id = batch
                logprobabilities = model.forward(x.to(device))
                loss = criterion(logprobabilities, y_true.to(device))
                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())
                field_ids_list.append(field_id)
        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list), torch.cat(field_ids_list)


def parse_args():
    parser = argparse.ArgumentParser(description='Train an evaluate time series deep learning models on the'
                                                 'DENETHOR dataset. This script trains a model on training dataset'
                                                 'partition, evaluates performance on a validation or evaluation partition'
                                                 'and stores progress and model paths in --logdir')
    parser.add_argument(
        'spatial_encoder', type=str, default="mobilenet_v3_small", help='select CNN model architecture. Available models '
                                                'correspond to torchvision models (https://pytorch.org/vision/stable/models.html)')
    parser.add_argument(
        'temporal_encoder', type=str, default="LSTM", help='select model architecture. Available models are '
                                                '"LSTM","TempCNN","MSRestNet","TransformerEncoder"')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=4, help='batch size (number of time series processed simultaneously)')
    parser.add_argument(
        '-i', '--imagesize', type=int, default=32, help='square size of the images. fields larger than the size will be cropped, smaller fields zero-padded')
    parser.add_argument(
        '-D', '--datapath', type=str, default="/ssd/DENETHOR/PlanetL3H/Test/PF-SR", help='PF-SR folder containg tiff images')
    parser.add_argument(
        '-L', '--labelgeojson', type=str, default="/ssd/DENETHOR/crops_test_2019.geojson",
        help='to the geojson containing label data')
    parser.add_argument(
        '-w', '--workers', type=int, default=0, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '-d', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". '
                                                       'default will check by torch.cuda.is_available() ')
    parser.add_argument(
        '-l', '--logdir', type=str, default="/tmp", help='logdir to store progress and models (defaults to /tmp)')
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args

def confusion_matrix_figure(conf_matrix, labels):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    cm = conf_matrix / (conf_matrix.sum(1) + 1e-12)
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel('Predictions', fontsize=18)
    ax.set_ylabel('Actuals', fontsize=18)
    ax.set_title('Confusion Matrix', fontsize=18)
    return fig

if __name__ == "__main__":
    args = parse_args()

    test(args)
