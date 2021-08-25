import torch
import os
import geopandas as gpd
import rasterio as rio
from rasterio import features
import numpy as np
import zipfile

CLASSES = ["Wheat", "Rye", "Barley", "Oats", "Corn", "Oil Seeds", "Root Crops", "Meadows", "Forage Crops"]
CROP_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, tifroot, labelgeojson, transform=None, min_area = 1000):
        self.tifroot = tifroot
        #self.npyfolder = os.path.abspath(os.path.join(self.tifroot,"..","npy"))
        # LK: make npy folder specific to input directory to avoid year conflicts
        self.npyfolder = os.path.abspath(self.tifroot+"_npy")
        tifs = [f for f in os.listdir(self.tifroot) if f.endswith(".tif")]
        self.tifs = sorted(tifs)
        self.labels = gpd.read_file(labelgeojson)

        self.data_transform = transform

        # read coordinate system of tifs and project labels to the same coordinate reference system (crs)
        with rio.open(os.path.join(self.tifroot, self.tifs[0])) as image:
            self.crs = image.crs
            self.transform = image.transform

        mask = self.labels.geometry.area > min_area
        print(f"ignoring {(~mask).sum()}/{len(mask)} fields with area < {min_area}m2")
        self.labels = self.labels.loc[mask]

        self.labels = self.labels.to_crs(self.crs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        feature = self.labels.iloc[item]

        npyfile = os.path.join(self.npyfolder,f"{feature.fid}.npz")
        if os.path.exists(npyfile): # use saved numpy array if already created
            try:
                object = np.load(npyfile)
                image_stack = object["image_stack"]
                mask = object["mask"]
            except zipfile.BadZipFile:
                print(f"{npyfile} is a bad zipfile...")
                raise
        else:
            left, bottom, right, top = feature.geometry.bounds

            window = rio.windows.from_bounds(left, bottom, right, top, self.transform)

            # reads each tif in tifs on the bounds of the feature. shape T x D x H x W
            image_stack = np.stack([rio.open(os.path.join(self.tifroot,tif)).read(window=window) for tif in self.tifs])

            # get meta data from first image to get the windowed transform
            with rio.open(os.path.join(self.tifroot,self.tifs[0])) as src:
                win_transform = src.window_transform(window)

            out_shape = image_stack[0,0].shape
            assert out_shape[0] > 0 and out_shape[1] > 0, f"fid:{feature.fid} image stack shape {image_stack.shape} is zero in one dimension"

            # rasterize polygon to get positions of field within crop
            mask = features.rasterize(feature.geometry, all_touched=True,
                                      transform=win_transform, out_shape=image_stack[0,0].shape)

            print(f"saving time series to {npyfile} for faster loading next time...")
            # save image stack as zipped numpy arrays for faster loading next time
            os.makedirs(self.npyfolder, exist_ok=True)
            np.savez(npyfile, image_stack=image_stack, mask=mask, feature=feature.drop("geometry").to_dict())

        if self.data_transform is not None:
            image_stack, mask = self.data_transform(image_stack, mask)

        return image_stack, CROP_IDS.index(feature.crop_id), mask * feature.fid
