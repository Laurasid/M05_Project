from preprocessor import preprocessing as pp
from model import linear_regression as lr
from model import regression_tree as rt
from analysis import analyse


if __name__ == "__main__":
    print(
        "\n***********************************\n"
        "M05 mini project \n"
        "Sidler Laura - Amos Jerome \n"
        "Wine quality and housing ML\n"
        "***********************************\n"
    )

    # -----------------------------------------------------
    # --- Choose dataset
    print(
        "Which dataset do you want ? \n"
        "\t 1-Red Wine\n"
        "\t 2-White Wine\n"
        "\t 3-Housing\n"
    )
    nDataset = int(input())
    url = ""
    if nDataset == 1:
        url = "Data/winequality-red.csv"
    elif nDataset == 2:
        url = "Data/winequality-white.csv"
    elif nDataset == 3:
        url = "Data/housing.data"
    else:
        raise Exception("No dataset for that entry")

    print("Importing dataset...\n")

    dataset = pp.import_dataset(url)

    # -----------------------------------------------------
    # --- choose split technique
    print(
        "Which seed value for split do you want ? \n"
        "\t 1-seed set at 30\n"
        "\t 2-seed set at 20\n"
        "\t 3-seed set at 10\n"
    )
    nSplit = int(input())

    # ----------------------------------------------------
    # --- choose normalisation technique
    print(
        "Which features scaling methode do you want ? \n"
        "\t 1-Standard scaling\n"
        "\t 2-Min-max scaling\n"
        "\t 3-PolynomialFeatures scaling\n"
        "\t 4-Normalization\n"
    )
    nNormalization = int(input())

    print("Preprocessing...")
    x_train, x_test, y_train, y_test = pp.preprocessing(dataset, nSplit, nNormalization)
    print("dataset preprocessed")

    # --------------------------------------------------
    # --- choose the model
    print("Which regression model do you want to use?\n" 
          "\t 1-Linear regression\n"
          "\t 2-Regression tree\n")
    nRegression = int(input())

    if nRegression == 1:
        regressor = lr.train(x_train, y_train)
        y_pred = lr.predict(regressor, x_test)

    if nRegression == 2:
        regressor = rt.train(x_train, y_train)
        y_pred = rt.predict(regressor, x_test)

    print("\n********* ANALYSE *********")
    mae = round(analyse.mae(y_test, y_pred),2)
    print("MAE  : \t", mae)
    r2 = round(analyse.r2(y_test, y_pred),2)
    print("R2   : \t", r2)
    rmse = round(analyse.rmse(y_test, y_pred),2)
    print("RMSE : \t", rmse)

    analyse.correlation_matrix(dataset)
