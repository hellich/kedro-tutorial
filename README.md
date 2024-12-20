﻿# kedro-tutorial

## prerequisites

- git
- micromamba


## setup
```
micromamba.exe create -n spaceflights310 python=3.10 -c conda-forge -y

micromamba activate C:\Users\33659\AppData\Roaming\mamba\envs\spaceflights310

pip install kedro

kedro info

kedro new #create a blank project

pip install notebook # install jupyter notebook

jupyter notebook # run jupyter notebook

pip install kedro-viz # kedro plugin for viz
```

## create kedro pipeline

```
cd C:\code\kedro-tutorial\spaceflights

kedro pipeline create data_processing
```

## list kedro pipelines

```
kedro registry list
```

## run kedro pipeline

```
kedro run --pipeline data_processing
```

# run kedro viz

```
kedro viz
```

runs web application locally

## run kedro with specific runner

```
kedro run --runner=ThreadRunner --pipeline=data_science #IO bound pipeline

kedro run --runner=ParallelRunner --pipeline=data_science  #CPU bound pipeline
```

## list kedro dataset factories

```
kedro catalog rank
```
```
kedro catalog resolve
```

## run pipeline on a specific environment

```
kedro run --pipeline=data_science --env=test
```

## to connect to external s3 

```
export FSSPEC_S3_ENDPOINT_URL=http://127.0.0.1:9000
export FSSPEC_S3_KEY=minioadmin
export FSSPEC_S3_SECRET=minioadmin
```

## create kedro package
```
kedro package
```

this will generate 2 files inside a "dist" folder : .whl (wheel python package that contains kedro pipelines) and .tar.gz (file with the configuration files)


```
micromamba install -c conda-forge blosc lzo hdf
```

Install all the dependencies with also the spaceflights package
```
pip install "C:\code\kedro-tutorial\spaceflights\dist\spaceflights-0.1-py3-none-any.whl"
```

Run kedro package
```
python -m spaceflights --conf-source="C:\code\kedro-tutorial\spaceflights\dist\conf-spaceflights.tar.gz" --env=test
```

## package kedro project into docker image

```
pip install kedro-docker

kedro docker init # will generate a dockerfile

kedro docker build --base-image python:3.10-slim #build docker image
```
