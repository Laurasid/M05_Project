import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, Normalizer
from sklearn.model_selection import train_test_split
"""
1. import dataset
2. take care of missing data
3. encode categorical data
4. split train test
5. feature scaling
"""


def import_dataset(url):
    #different file extension : .csv and .data
    fileExtension = url.split('.')[1]
    #different treatment for csv and for .data
    if (fileExtension == 'csv'):
        dataset = pd.read_csv(url, sep=';')
    elif (fileExtension == 'data'):
        #load text file
        text = np.loadtxt(url)
        #turn text file as pandas dataframe
        dataset = pd.DataFrame(text, columns=['CRIM','ZN','INDUS', 'CHAS','NOX', 'RM','AGE','DIS','RAD','TAX', 'PTRATIO','B','LSTAT', 'MEDV'])
    else:
        raise Exception("This dataset extension cannot be use. Use .csv or .data file.")
    return dataset



# Normalize the data with standard scaling
###
def standardScaling(dataset):
    scaler = StandardScaler()
    result = scaler.fit_transform(dataset)
    return result
###
# Normalize with min-max scaler
###
def minMaxScaling(dataset):
    scaler = MinMaxScaler()
    result = scaler.fit_transform(dataset)
    return result

###
# Normalize with polynomial scaler
###
def polynomialScaling(dataset):
    scaler = PolynomialFeatures()
    result = scaler.fit_transform(dataset)
    return result

###
#
###
def normalize(dataset):
    scaler = Normalizer()
    result = scaler.fit_transform(dataset)
    return result


###
#
###
def preprocessing(dataset, nSplit, nNorm):
    data = dataset.dropna(axis='index')
    data = data.drop_duplicates()

    x = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    #split into train-test set
    np.random.seed(0)
    if nSplit == 1:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=3)
    elif nSplit == 2:
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state = 2)
    elif nSplit == 3:
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state = 1)
    else:
        raise Exception("Nothing developped for this split entry. Choose 1,2 or 3 !")

    #normalize the values x_train, x_test
    if nNorm == 1:
        x_train = standardScaling(x_train)
        x_test = standardScaling(x_test)
    elif nNorm ==2:
        x_train = minMaxScaling(x_train)
        x_test = minMaxScaling(x_test)
    elif nNorm ==3:
        x_train = polynomialScaling(x_train)
        x_test = polynomialScaling(x_test)
    elif nNorm ==4:
        x_train = normalize(x_train)
        x_test = normalize(x_test)
    else:
        raise Exception("Nothing developped for this scaling entry. Choose 1,2,3 or 4!")


    return x_train,x_test,y_train, y_test


