import matplotlib.pyplot as plt
import numpy as np
from preprocessor import  preprocessing as pp
from model import linear_regression as lr
from model import ann as ann
from analysis import analyse

import seaborn as sns


if __name__ == "__main__":
    print("\n***********************************\n"
          "M05 mini project \n"
          "Sidler Laura - Amos Jerome \n"
          "Wine quality and housing ML\n"
          "***********************************\n")


    #--- Choose dataset
    print("Which dataset do you want ? \n"
          "\t 1-Red Wine\n"
          "\t 2-White Wine\n"
          "\t 3-Housing\n")
    nDataset = int(input())
    url=""
    if nDataset == 1 :
        url='Data/winequality-red.csv'
    elif nDataset == 2:
        url = 'Data/winequality-white.csv'
    elif nDataset == 3:
        url = 'Data/housing.data'
    else:
        raise Exception("No dataset for that entry")

    print("Importing dataset...\n")

    dataset = pp.import_dataset(url)
    #print("dataset brut")
    #print(dataset)

    #-----------------------------------------------------
    # --- chose split technique
    print("Which seed value for split do you want ? \n"
          "\t 1-seed set at 3\n"
          "\t 2-seed set at 2\n"
          "\t 3-seed set at 1\n")
    nSplit = int(input())

    #----------------------------------------------------
    #--- chose normalisation technique
    print("Which features scaling methode do you want ? \n"
          "\t 1-Standard scaling\n"
          "\t 2-Min-max scaling\n"
          "\t 3-PolynomialFeatures scaling\n"
          "\t 4-Normalization\n")
    nNormalization = int(input())



    print("Preprocessing...")
    x_train, x_test, y_train,y_test = pp.preprocessing(dataset,nSplit, nNormalization)
    print("dataset preprocessed")

    #--------------------------------------------------

    print("Which regression model do you want to use?\n"
          "\t 1-Linear regression\n")
    nRegression = int(input())

    if(nRegression == 1):
        regressor = lr.train(x_train, y_train)
        y_pred = lr.predict(regressor, x_test)


    print("\n********* ANALYSE *********")
    mae = analyse.mae(y_test, y_pred)
    print("MAE  : \t", mae)
    r2 = analyse.r2(y_test, y_pred)
    print("R2   : \t", r2)
    rmse = analyse.rmse(y_test, y_pred)
    print("RMSE : \t", rmse)


    analyse.correlation_matrix(dataset)
