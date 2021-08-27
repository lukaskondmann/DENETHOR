# DENETHOR - A dataset for crop type mapping from daily, analysis-ready satellite data


## Download

```
Shared privately with the reviewers
```

## Requirements 

```
pip install -U -r requirements.txt
```

## Training

```
python train.py 
    mobilenet_v3_small #Spatial Encoder. If "none" => pixel average
    transformermodel #Temporal Encoder
    --logdir /tmp 
    --datapath /ssd/DENETHOR/PlanetL3H/Train/PF-SR 
    --labelgeojson /ssd/DENETHOR/crops_train_2018.geojson
    --batchsize 12 --workers 8
```

## Test a model 
requires a model checkpoint in the log directory
```
python test.py 
    mobilenet_v3_small #Spatial Encoder. If "none" => pixel average
    transformermodel #Temporal Encoder
    --logdir /tmp 
    --datapath /ssd/DENETHOR/PlanetL3H/Test/PF-SR 
    --labelgeojson /ssd/DENETHOR/crops_test_2019.geojson
    --batchsize 24
```


## Train/Test PSETAE
seperate script for testing & training PSETAE models
```
python test_psetae.py 
    --modelname psetae 
    --logdir /tmp 
    --datapath /ssd/DENETHOR/PlanetL3H/Test/PF-SR 
    --labelgeojson /ssd/DENETHOR/crops_test_2019.geojson
    --batchsize 24
```
## RF Models with S1/S2

In case you want to run this, adjust the respective paths in random_forest.py

## Tests

test all spatio-temporal model combinations
```angular2html
pytest tests.py
```

test single configuration (e.g. resnet18 + lstm)
```
pytest tests.py -k "test_spatiotemporalmodel[resnet18-lstm]"
```
