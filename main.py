import numpy as np

from preprocessor import  preprocessing as pp
from model import linear_regression as lr
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
    #print("ytest : ", x_test.shape)


    #start LINEAR REGRESSION

    regressor = lr.train(x_train, y_train)

    y_pred = lr.predict(regressor, x_test)

    #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    sum = 0
    for i in range (len(y_pred)):
        sum += (y_pred[i]-y_test[i])
    error = sum/(len(y_pred))
    print (error)

    plt.plot(y_pred,'ro')
    plt.plot(y_test,'ro', color='green')
    plt.show()