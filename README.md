# kedro-tutorial

## prerequisites

- git
- micromamba


## setup
```
micromamba.exe create -n spaceflights310 python=3.10 -c conda-forge -y

micromamba activate C:\Users\33659\AppData\Roaming\mamba\envs\spaceflights310

pip install kedro

kedro info

kedro new #create a black project

pip install notebook # install jupyter notebook

jupyter notebook # run jupyter notebook

pip install kedro-viz # kedro plugin for viz
```

## create kedro pipelineù

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