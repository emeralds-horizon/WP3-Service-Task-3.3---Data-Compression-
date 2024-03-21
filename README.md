# Description
This repository presents the initial version of the Data Fusion service, developed under 3.3 of Work Package 3 within the EMERALDS project, with a primary focus on estimating missing values. Graph Convolutional Neural Networks (GCNNs) emerge as pivotal tools for this task, excelling in capturing spatial correlations within graphs. The module integrates various GCNN methods, encompassing spatial pattern extractor layers employing spatial-based and spectral-based techniques, alongside temporal pattern extractor layers with and without attention mechanisms. These components collaborate synergistically to estimate missing values.

# Requirements
- Python >= 3.7
- NumPy
- Torch
- Pandas

# Data input/output structures 
Datasets containd two files:
1. Structure of the transportation network:
   Shape (Origin sensor ID, Destination sensor ID, Cost (optianal))
   This file is in CSV format and contains the conneectivity of sensor collcetors in the network.
2. Historical data:
   Shape (temporal sequence length, number of sensors, features)
   This file is in NDZ format and contains historical data of each data collector with probability of having missing values. Currently only "speed" feature is using for estimation task. 

The output file is containing the historical data with estimated values for missings. 



# Deployment


# Usage - Executing program
Set config files and then train the model.
```
python train.py
```

# Authors
ULB
