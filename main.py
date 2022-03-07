import numpy as np
from preprocessor import  preprocessing as pp
from model import linear_regression as lr
from analysis import analyse
import matplotlib.pyplot as plt


if __name__ == "__main__":
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
    #check some size
    #print(dataset.shape)
    #print("xtrain : ", x_train.shape)
    #print("xtest : ", x_test.shape)
    #print("ytrain : ", y_train.shape)
    #print("ytest : ", y_test.shape)


    #Test LINEAR REGRESSION

    regressor = lr.train(x_train, y_train)

    y_pred = lr.predict(regressor, x_test)


    mae = analyse.MAE(y_test, y_pred)
    print("mae : ", mae)
    r2 = analyse.r2(y_test, y_pred)
    print("r2 : ", r2)
    rmse = analyse.rmse(y_test, y_pred)
    print("rmse : ", rmse)