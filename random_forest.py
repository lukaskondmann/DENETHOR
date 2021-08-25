from s2dataset import S2DatasetV2, S2Dataset
from dataset import Dataset, CLASSES
from s1dataset import S1Dataset
import numpy as np
import torch
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

show_test_results = True # set to true after hyperparameter tuning

BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
PLANETBANDS = ["blue","green","red","nir"]

DATAPATHS = {
    "s2":{
        "traindata":"/ssd/DENETHOR/S2/Train/s2_train.zip",
        "testdata":"/ssd/DENETHOR/S2/Test/s2_test.zip"
    },
    "s1-asc": {
        "traindata":"/ssd/DENETHOR/S1/Train/s1_train_asc.zip",
        "testdata":"/ssd/DENETHOR/S1/Test/s1_test_asc.zip"
    },
    "s1-des": {
        "traindata":"/ssd/DENETHOR/S1/Train/s1_train_des.zip",
        "testdata":"/ssd/DENETHOR/S1/Test/s1_test_des.zip"
    },
    "planet": {
        "traindata":"/ssd/DENETHOR/PlanetL3H/Train/PF-SR",
        "testdata":"/ssd/DENETHOR/PlanetL3H/Test/PF-SR"
    }
}

LABELDATAPATHS = {
    "traindata": "/ssd/DENETHOR/crops_test_2019.geojson",
    "testdata": "/ssd/DENETHOR/crops_train_2018.geojson"
}
mode = "test"

datasets = ["s2","planet"] #  "s1-asc","s1-des" "s2", "planet"

def main():

    X_train, y_train, fid_train, X_test, y_test, fid_test = get_data(datasets)


    print(f"{datasets}. with {X_train.shape[1]} features")

    clf = RandomForestClassifier(random_state=0)
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier
    #clf = HistGradientBoostingClassifier(random_state=0)

    if mode=="validation":
        N = X_train.shape[0]
        indices = list(range(N))
        np.random.RandomState(0).shuffle(indices)
        split = int(np.floor(0.25 * N))
        train_indices, val_indices = indices[split:], indices[:split]

        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        fid_val = fid_train[val_indices]

        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        fid_train = fid_train[train_indices]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        print("val data")
        print(classification_report(y_val, y_pred, labels=np.arange(8), target_names=CLASSES))
    elif mode=="test":
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("testdata")
        print(classification_report(y_test, y_pred, labels=np.arange(8), target_names=CLASSES))

def get_data(data):
    """
    a wrapper around features to merge features of different datasets.
    data is a list of datasets e.g.
    ["s1-asc", "s1-desc", "s2"]
    """

    def zip_features(X1, y1, fid1, X2, y2, fid2):
        """
        joins two sets X,y,id arrays by same id
        """
        lfid1 = list(fid1)
        lfid2 = list(fid2)

        X = []
        y = []
        fid = []
        for f in fid1:
            if f not in lfid1:
                print(f"fid {f} not in first dataset. skipping...")
            if f not in lfid2:
                print(f"fid {f} not in second dataset. skipping...")
                continue
            idx1 = lfid1.index(f)
            idx2 = lfid2.index(f)
            X.append(np.hstack([X1[idx1], X2[idx2]]))

            assert y1[idx1] == y2[idx2]
            y.append(y1[idx1])
            assert fid1[idx1] == fid2[idx2]
            fid.append(fid1[idx1])

        return np.stack(X), np.stack(y), np.stack(fid)

    X_train, y_train, fid_train, X_test, y_test, fid_test = features(data[0])
    print(
        f"added {data[0]} data: {X_train.shape[1]} features in train {X_train.shape[0]}, test {X_test.shape[0]} samples")
    for d in data[1:]:
        X_train_, y_train_, fid_train_, X_test_, y_test_, fid_test_ = features(d)
        X_train, y_train, fid_train = zip_features(X_train, y_train, fid_train, X_train_, y_train_, fid_train_)
        X_test, y_test, fid_test = zip_features(X_test, y_test, fid_test, X_test_, y_test_, fid_test_)

        print(f"added {d} data: {X_train.shape[1]} features in train {X_train.shape[0]}, test {X_test.shape[0]} samples")
    return X_train, y_train, fid_train, X_test, y_test, fid_test

def features(data):
    testlabel = LABELDATAPATHS["traindata"]
    labelgeojson = LABELDATAPATHS["testdata"]


    # an intermediate cached array of the dataset for faster loading
    featuresnpz = f"/tmp/{data}.npz"

    if os.path.exists(featuresnpz):
        print(f"{featuresnpz} exists. loading from there. delete to rebuild features")
        npz = np.load(featuresnpz)
        X_train = npz["X_train"]
        X_test = npz["X_test"]
        y_train = npz["y_train"]
        y_test = npz["y_test"]
        fid_train = npz["fid_train"]
        fid_test = npz["fid_test"]

        return X_train, y_train, fid_train, X_test, y_test, fid_test
    else:

        if data == "s2":
            datapath = DATAPATHS["s2"]["traindata"]
            testdata = DATAPATHS["s2"]["testdata"]

            def ndvi(X):
                red = X[:,BANDS.index("B04")]
                nir = X[:,BANDS.index("B08")]
                return (nir-red) / (nir + red)

            def transform(X, cld):
                msk = cld < 40
                X = X * 1e-4

                # add NDVI feature
                n = ndvi(X)
                X = np.hstack([X, n[:, None]])

                X_ = X[msk[:, 0], :]

                # get min max std of each cloud filtered band
                feat = [f(X_, axis=0) for f in [np.max, np.min, np.median, np.mean, np.std]]

                # reconstruct index of the max on the original not-cloud filtered data
                t_max = np.argmax(X == feat[0], axis=0)
                t_max = t_max / X.shape[0]

                # reconstruct index of the min on the original not-cloud filtered data
                t_min = np.argmax(X == feat[1], axis=0)
                t_min = t_min / X.shape[0]

                # make one long feature vector again
                X = np.stack(feat).reshape(-1)

                # add time index of min and max values
                X = np.hstack([X,t_min, t_max])

                return X

            dataset = S2Dataset(zippath=datapath,
                              labelgeojson=labelgeojson,
                              transform=transform)

            testdataset = S2Dataset(zippath=testdata,
                                    labelgeojson=testlabel,
                                    transform=transform)

        elif data == "planet":
            datapath = DATAPATHS["planet"]["traindata"]
            testdata = DATAPATHS["planet"]["testdata"]

            def ndvi(X):
                red = X[:,PLANETBANDS.index("red")]
                nir = X[:,PLANETBANDS.index("nir")]
                return (nir-red) / (nir + red)

            def transform(image_stack, mask):
                bag_of_pixels = image_stack.reshape(365,4,-1)[:,:,mask.reshape(-1)]
                X = bag_of_pixels.mean(-1)

                X = X * 1e-4

                # add ndvi as band
                n = ndvi(X)
                X = np.hstack([X, n[:, None]])

                # get min max std of each cloud filtered band
                feat = [f(X, axis=0) for f in [np.max, np.min, np.median, np.mean, np.std]]

                # reconstruct index of the max on the original not-cloud filtered data
                t_max = np.argmax(X == feat[0], axis=0)
                t_max = t_max / X.shape[0]

                # reconstruct index of the min on the original not-cloud filtered data
                t_min = np.argmax(X == feat[1], axis=0)
                t_min = t_min / X.shape[0]

                # make one long feature vector again
                X = np.stack(feat).reshape(-1)

                # add time index of min and max values
                X = np.hstack([X,t_min, t_max])

                return X, 1

            dataset = Dataset(tifroot=datapath, labelgeojson=labelgeojson, transform=transform)
            testdataset = Dataset(tifroot=testdata,
                                    labelgeojson=testlabel,
                                    transform=transform)

        elif "s1" in data:
            if "s1-des" in data:
                datapath = DATAPATHS["s1-des"]["traindata"]
                testdata = DATAPATHS["s1-des"]["testdata"]
            elif "s1-asc" in data:
                datapath = DATAPATHS["s1-asc"]["traindata"]
                testdata = DATAPATHS["s1-asc"]["testdata"]


            def transform(X):
                #X = X * 1e-4
                X = np.nan_to_num(X, neginf=0, posinf=0)

                vv = X[:,0]
                vh = X[:,1]

                vvvh = vv/(vh + 1e-12)
                vvvh -= vvvh.min()
                vvvh /= (vvvh.max() + 1e-12)

                X = np.hstack([X, vvvh[:, None]])

                # get min max std of each cloud filtered band
                feat = [f(X, axis=0) for f in [np.max, np.min, np.median, np.mean, np.std]]

                # reconstruct index of the max on the original not-cloud filtered data
                t_max = np.argmax(X == feat[0], axis=0)
                t_max = t_max / X.shape[0]

                # reconstruct index of the min on the original not-cloud filtered data
                t_min = np.argmax(X == feat[1], axis=0)
                t_min = t_min / X.shape[0]

                # make one long feature vector again
                X = np.stack(feat).reshape(-1)

                # add time index of min and max values
                X = np.hstack([X,t_min, t_max])

                assert not np.isnan(X).any()

                return X

            dataset = S1Dataset(zippath=datapath,
                              labelgeojson=labelgeojson,
                              transform=transform)

            testdataset = S1Dataset(zippath=testdata,
                                    labelgeojson=testlabel,
                                    transform=transform)

        #traindataset = torch.utils.data.Subset(dataset, train_indices)
        #valdataset = torch.utils.data.Subset(dataset, val_indices)

        def iterate_dataset(dataset):
            X = []
            y = []
            fid = []
            for x_, y_, fid_ in tqdm(dataset):
                X.append(x_)
                y.append(y_)
                fid.append(fid_)

            X = np.array(X)
            y = np.array(y)
            fid = np.array(fid)
            return X,y,fid


        X_train,y_train,fid_train = iterate_dataset(dataset)
        X_test,y_test,fid_test = iterate_dataset(testdataset)

        print(f"saving {featuresnpz}")
        np.savez(featuresnpz,
                 X_train=X_train,
                 X_test=X_test,
                 y_train=y_train,
                 y_test=y_test,
                 fid_train=fid_train,
                 fid_test=fid_test)

        return X_train, y_train, fid_train, X_test, y_test, fid_test

if __name__ == '__main__':
    main()
