# DENETHOR - A dataset for crop type mapping from daily, analysis-ready satellite data


Recent advances in remote sensing products allow near-real time monitoring of the
Earth's surface. Despite increasing availability of near-daily time-series of satellite
imagery, there has been little exploration of deep learning methods to utilize
the unprecedented temporal density of observations. This may be particularly
interesting in crop monitoring where time-series remote sensing data has been used
frequently to exploit phenological differences of crops in the growing cycle over
time. Therefore, we introduce **DENETHOR**: The **D**ynamic**E**arth**NET** dataset for
**H**armonized, inter-**O**perabel, analysis-**R**eady, daily crop monitoring from space.
Our dataset contains daily, analysis-ready Planet Fusion data together with Sentinel-
1 and 2 time-series for crop type classification in Brandenburg, Germany. Our
baseline experiments underline that incorporating the available spatial and temporal
information fully may not be straightforward and could require the design of
tailored architectures. The dataset presents two main challenges to the community:
Exploit the temporal dimension for improved crop classification and ensure that
models can handle a domain shift to a different year.

![](33N-17E-243N_compressed.gif)

If you use this dataset, please cite:

```
@article{kondmann2021denethor,
  title={DENETHOR: The DynamicEarthNET dataset for Harmonized, inter-Operable, analysis-Ready, daily crop monitoring from space},
  author={Kondmann, Lukas and Toker, Aysim and Ru{\ss}wurm, Marc and Camero, Andr{\'e}s and Peressuti, Devis and Milcinski, Grega and Mathieu, Pierre-Philippe and Long{\'e}p{\'e}, Nicolas and Davis, Timothy and Marchisio, Giovanni and others},
  year={2021}
}
```



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
