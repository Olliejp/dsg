# DSG Sept 2020 - Strathclyde

The overall aim of this challenge is to investigate the spatial and temporal distribution of failures across the power grid, examine the relationships between the components and predict the first failure before it happens in order to prevent it. We use the dataset consisting of 44,064 power grid simulations provided by the University of Strathclyde \& Supergen Energy Networks Hub and utilise machine learning methods to achieve exciting results. We also aim to identify the limitations of the data and the methods used in order to provide a road map for future research.

## Project Structure

    .
    ├── trained_models/                 # Trained models
    ├── docs/                           # Documentation and guidelines
    ├── data/                           # Schematic data, results
    ├── notebooks/                      # Jupyter notebooks
    ├── scripts/                        # Model code
    ├── Strathclyde_DSG_presentation    # Short presentation of the work done
    ├── requirements.txt                # Dependencies
    └── README.md

## Installation
...

## Running Modules

    .
    ├── window_dataset.py               # function to create windowed input output pairs from timeseries dataset
    ├── utils.py                        # functions useful for the data at hand
    ├── manipulation.py                 # functions to load, preprocess and save data           
    ├── cleaning.py                     # functions for preprocessing
    └── __init__.py

## Final presentation link

https://docs.google.com/presentation/d/17FKMYA4Gb8Scv0Qe-CfMjcWFrKCB8-3S9TovyL_ft0Q/edit#slide=id.p

## Future Work

Considering the duration of the data study group, many challenges could not be investigated, but some suggestions are proposed to tackle these challenges in more depth. Promising approaches for future work are identified below.

- One could change the time-frame analysed and investigate the corresponding impact on the performances of the models.
- Performing dimensionality reduction based on feature importance, correlation and clustering may help to reduce the computational requirements for model training. 
- Multi-class classification algorithms could be implemented and tested. For instance graph based models may be a fruitful avenue to explore due to the spatio-temporal nature of the problem - we know the network topology.
- Cascading sequence forecasting: predict the time steps associated with a cascade given a specified target forecasting window size. 
- Multi-class classification: Cascading sequences are likely spatially correlated. Features which allow the model to learn network topology should be identified. This could be achieved by evaluating model performance on new simulated data with modifications made to the network (taking a line out of service, taking a generator offline). The dataset formulated as a multi-class classification problem will be highly imbalanced, with a significant number of minority classes. A number of approaches should be investigated to attempt to mitigate this issue which include: finding optimal window data sizes and window shift values, reducing the amount of stable data the model is trained on, upsampling of minority classes using synthetic methods such as SMOTE or upsampling of minority classes using synthetic data generated from autoencoders (using active dropout layers for data reconstruction will provide stochastic outputs).

## Contributors

Maryleen Ndubuaku, Oliver Paul, Domenic Di Francesco, Josh Nevin, Alara Dirik, Chloé Sekkat, Amarsagar Reddy Ramapuram Matavalam.
