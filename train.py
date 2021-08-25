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



def train(args):
    
    # MR: setting this to False prevents CUDNN_STATUS_EXECUTION_FAILED on my local GeForce GTX 970
    # may speed up training if set to true.
    # LK: can now be passed as an argument, defaults to false
    torch.backends.cudnn.enabled=args.backend
    torch.backends.cudnn.benchmark = True
    
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

            # rotations
            rot = np.random.choice([0, 1, 2, 3])
            image_stack = np.rot90(image_stack, rot, [2, 3])
            mask = np.rot90(mask, rot)

            # flip up down
            if np.random.rand() < 0.5:
                image_stack = np.flipud(image_stack)
                mask = np.flipud(mask)

            # flip left right
            if np.random.rand() < 0.5:
                image_stack = np.fliplr(image_stack)
                mask = np.fliplr(mask)

        image_stack = image_stack * 1e-4

        # z-normalize
        image_stack -= 0.1014 + np.random.normal(scale=0.01)
        image_stack /= 0.1171 + np.random.normal(scale=0.01)

        return torch.from_numpy(np.ascontiguousarray(image_stack)).float(), torch.from_numpy(np.ascontiguousarray(mask))

    dataset = Dataset(tifroot=args.datapath,
                      labelgeojson=args.labelgeojson,
                      transform=transform)

    indices = list(range(len(dataset)))
    np.random.RandomState(0).shuffle(indices)
    split = int(np.floor(args.validation_split * len(dataset)))
    train_indices, val_indices = indices[split:], indices[:split]

    traindataset = torch.utils.data.Subset(dataset, train_indices)
    valdataset = torch.utils.data.Subset(dataset, val_indices)

    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batchsize, num_workers=args.workers)
    testdataloader = torch.utils.data.DataLoader(valdataset, batch_size=args.batchsize, num_workers=args.workers)

    device = torch.device(args.device)
    model = SpatiotemporalModel(spatial_backbone=args.spatial_encoder, temporal_backbone=args.temporal_encoder,
                                input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength, device=device)
    model.to(device)

    # loss becomes nan sometimes during training. trying gradient clipping
    clip_value = 1e-2
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Initialized {model.modelname}")

    logdir = os.path.join(args.logdir, model.modelname)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging results to {logdir}")

    summarywriter = SummaryWriter(log_dir=logdir)

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    log = list()
    start_epoch = 1
    # resume model if exists
    snapshot_path = os.path.join(logdir, "model.pth.tar")
    if os.path.exists(snapshot_path):
        checkpoint = torch.load(snapshot_path)
        start_epoch = checkpoint["epoch"]
        log = checkpoint["log"]
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        model.load_state_dict(checkpoint["model_state"])
        print(f"resuming from {snapshot_path}, epoch {start_epoch}")

    for epoch in range(start_epoch,args.epochs+1):
        train_loss = train_epoch(model, optimizer, criterion, traindataloader, device)
        test_loss, y_true, y_pred, *_ = test_epoch(model, criterion, testdataloader, device)
        scores = metrics(y_true.cpu(), y_pred.cpu())
        scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
        test_loss = test_loss.cpu().detach().numpy()[0]
        train_loss = train_loss.cpu().detach().numpy()[0]

        scores["epoch"] = epoch
        scores["trainloss"] = train_loss
        scores["testloss"] = test_loss
        log.append(scores)

        summarywriter.add_scalars("losses", dict(train=train_loss,
                                                 test=test_loss), global_step=epoch)
        summarywriter.add_scalars("metrics",
                                  {key: scores[key] for key in
                                   ['accuracy', 'kappa', 'f1_micro', 'f1_macro', 'f1_weighted', 'recall_micro',
                                    'recall_macro', 'recall_weighted', 'precision_micro', 'precision_macro',
                                    'precision_weighted']},
                                  global_step=epoch)

        confusion_matrix = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred.cpu().detach().numpy(),
                                                            labels=np.arange(len(CLASSES)))
        summarywriter.add_figure("confusion_matrix",confusion_matrix_figure(confusion_matrix, labels=CLASSES),
                                 global_step=epoch)

        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(os.path.join(logdir, "trainlog.csv"))

        torch.save(
            dict(
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                epoch=epoch + 1,
                log=log),
            snapshot_path)
        if len(log) > 2:
            if test_loss < np.array([l["testloss"] for l in log[:-1]]).min():
                best_model = snapshot_path.replace("model.pth.tar","model_best.pth.tar")
                print(f"new best model with testloss {test_loss:.2f} at {best_model}")
                shutil.copy(snapshot_path, best_model)

        print(f"epoch {epoch}: trainloss {train_loss:.2f}, testloss {test_loss:.2f} " + scores_msg)


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


def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    losses = list()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, batch in iterator:
            optimizer.zero_grad()
            x, y_true, _ = batch
            loss = criterion(model.forward(x.to(device)), y_true.to(device))
            loss.backward()
            optimizer.step()
            iterator.set_description(f"train loss={loss:.2f}")
            losses.append(loss)
    return torch.stack(losses)


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
                                                 'BreizhCrops dataset. This script trains a model on training dataset'
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
        '-e', '--epochs', type=int, default=150, help='number of training epochs (training on entire dataset)')
    parser.add_argument(
        '-D', '--datapath', type=str, default="/ssd/DENETHOR/PlanetL3H/Train/PF-SR", help='PF-SR folder containg tiff images')
    parser.add_argument(
        '-L', '--labelgeojson', type=str, default="/ssd/DENETHOR/crops_train_2018.geojson",
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
        '-ba', '--backend', type=bool, default=False, help='This sets torch.backends.cudnn.enabled which may speed up training if true')
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

    train(args)
