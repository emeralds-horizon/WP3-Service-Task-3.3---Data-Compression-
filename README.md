# Description
This repository presents the initial version of the Data Fusion service, developed under 3.3 of Work Package 3 within the EMERALDS project, with a primary focus on estimating missing values. Graph Convolutional Neural Networks (GCNNs) emerge as pivotal tools for this task, excelling in capturing spatial correlations within graphs. The module integrates various GCNN methods, encompassing spatial pattern extractor layers employing spatial-based and spectral-based techniques, alongside temporal pattern extractor layers with and without attention mechanisms. These components collaborate synergistically to estimate missing values.

# Requirements
- Python >= 3.7
- NumPy
- Torch
- Pandas

# Data input/output structures 
Datasets comprise two files:
1. Transportation Network Structure:
   Shape (Origin sensor ID, Destination sensor ID, Cost (optianal))
   This CSV-formatted file delineates the connectivity of sensor collectors in the network.
2. Historical data:
   Shape (temporal sequence length, number of sensors, features)
   This NDZ-formatted file captures historical data with probability of missing values, from each data collector. At present, only the "speed" feature is utilized for this task.

The output file contains the historical data supplemented with estimated values for missings. 

# Deployment


# Usage - Executing program
Set config files and then train the model.
```
python train.py
```

# Authors
ULB
