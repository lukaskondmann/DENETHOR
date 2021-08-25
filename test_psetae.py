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
from train import train_epoch, test_epoch
from models import PseLTae, PseTae

def test(args):

    num_classes = 9
    ndims = 4
    sequencelength = 365
    N_pixels = 20

    def transform(image_stack, mask):

        set_of_pixels = image_stack[:, :, mask > 0]
        mask = -1 # mask is meaninless now since we only consider pixels within a field boundary

        N_pixels_of_field = set_of_pixels.shape[2]

        # choose N_pixels pixels randomly
        idxs = np.random.randint(low=0,high=N_pixels_of_field,size=N_pixels)
        set_of_pixels = set_of_pixels[:, :, idxs]

        set_of_pixels = set_of_pixels * 1e-4

        # z-normalize
        set_of_pixels -= 0.1014 + np.random.normal(scale=0.01)
        set_of_pixels /= 0.1171 + np.random.normal(scale=0.01)

        return torch.from_numpy(np.ascontiguousarray(set_of_pixels)).float(), torch.from_numpy(np.ascontiguousarray(mask))

    dataset = Dataset(tifroot=args.datapath,
                      labelgeojson=args.labelgeojson,
                      transform=transform)

    if args.validation == True: 
        indices = list(range(len(dataset)))
        np.random.RandomState(0).shuffle(indices)
        split = int(np.floor(args.validation_split * len(dataset)))
        train_indices, val_indices = indices[split:], indices[:split]
        valdataset = torch.utils.data.Subset(dataset, val_indices)
        testdataloader = torch.utils.data.DataLoader(valdataset, batch_size=args.batchsize, num_workers=args.workers)
    else: 
        testdataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, num_workers=args.workers)
    modelname = args.modelname
    if modelname == "pseltae":
        model = PseLTae(input_dim=4, mlp1=[4, 32, 64], pooling='mean_std', mlp2=[128, 128], with_extra=False,
                     extra_size=0,
                     n_head=16, d_k=8, d_model=256, mlp3=[256, 128], dropout=0.2, T=365, len_max_seq=365, positions=None,
                     mlp4=[128, 64, 32, num_classes], return_att=False)
    elif modelname == "psetae":
        model = PseTae(input_dim=4, mlp1=[4, 32, 64], pooling='mean_std', mlp2=[128, 128], with_extra=False,extra_size=0,
                 n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000,
                       len_max_seq=365, positions=None, mlp4=[128, 64, 32, 20], return_att=False)  
    else:
        raise ValueError("provide as modelname either PseTae or PseLTae")    
    
    device = torch.device(args.device)
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train an evaluate time series deep learning models on the'
                                                 'DENETHOR dataset. This script trains a model on training dataset'
                                                 'partition, evaluates performance on a validation or evaluation partition'
                                                 'and stores progress and model paths in --logdir')
    parser.add_argument('--modelname', type=str, default="PseLTae", help='modelvariant PseTae or PseLTae')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=4, help='batch size (number of time series processed simultaneously)')
    parser.add_argument(
        '-i', '--imagesize', type=int, default=32, help='square size of the images. fields larger than the size will be cropped, smaller fields zero-padded')
    parser.add_argument(
        '-e', '--epochs', type=int, default=150, help='number of training epochs (training on entire dataset)')
    parser.add_argument(
        '-D', '--datapath', type=str, default="/ssd/DENETHOR/PlanetL3H/Test/PF-SR", help='PF-SR folder containg tiff images')
    parser.add_argument(
        '-L', '--labelgeojson', type=str, default="/ssd/DENETHOR/crops_test_2019.geojson",
        help='to the geojson containing label data')
    parser.add_argument(
        '-w', '--workers', type=int, default=0, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '--weight-decay', type=float, default=1e-6, help='optimizer weight_decay (default 1e-6)')
    parser.add_argument(
        '--learning-rate', type=float, default=1e-3, help='optimizer learning rate (default 1e-3)')
    parser.add_argument(
        '--validation_split', type=float, default=0.25, help='ratio of validation a to training data')
    parser.add_argument(
        '-d', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". '
                                                       'default will check by torch.cuda.is_available() ')
    parser.add_argument(
        '-v', '--validation', type=bool, default=False, help='Only pick the validation section of the data')
                                                                                                           
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
