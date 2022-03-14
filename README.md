# M05_Project


# Datasets
For this project we use three different dataset. The datasets are allready imported into project under the folder data.
  - Two dataset for wine quality (red-white) that could be found here : https://archive.ics.uci.edu/ml/datasets/wine+quality
  - One dataset for housing price that can be found here : https://archive.ics.uci.edu/ml/machine-learning-databases/housing/ 

# Installation
1. check the dependencies versions : 
  - Python : 3.8.12
  - Pandas : 1.4.1
  - Numpy : 1.22.2
  - Matplotlib : 3.5.1
  - Sklearn : 1.0.2
  - seaborn : 0.11.2

2. Open a terminal
3. Go into the dedicated folder
4. Run main.py : python main.py
5. Follow the instructions:
  - Choose the dataset
  - Choose the seed value for splitting dataset
  - Choose the scaling feature method

# Project structure
This project is composed of 3 packages and one main. 
  - Data : package containing the datasets and data info
  - Preprocessor : package containing the preprocessing methods
  - Model : package containing the differents regression models
  - main.py : main function of test program
  
```bash
|--M05_Project
    |
    |---Data
    |     |-housing.data
    |     |-housing.names
    |     |-Index
    |     |-winequality.names
    |     |-winequality-red.csv
    |     |-winequality-white.csv
    |
    |---preprocessor
    |     |-preprocessing.py
    |
    |---model
    |     |-linear_regression.py
    |     |-regression_tree.py
    |
    |---analysis
    |     |-analyse.py
    |     |-correlationMatrix.png
    |
    |-main.py
    |-README.md
```
