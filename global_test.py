from preprocessor import preprocessing as pp
from model import linear_regression as lr
from model import regression_tree as rt
from analysis import analyse


def run_all_possibilities():
    """
    Function that run all possibilities of dataset-split-scaling-algorithms

    :return: returnArray
         return an array of string containing all values
    """
    returnArray = []
    datasets = []

    #import the 3 datasets
    dataset1 = pp.import_dataset("Data/winequality-red.csv")
    dataset2 = pp.import_dataset("Data/winequality-white.csv")
    dataset3 = pp.import_dataset("Data/housing.data")

    #put datasets into an array
    datasets.append(("red_Wine",dataset1))
    datasets.append(("white_wine",dataset2))
    datasets.append(("housing",dataset3))

    #loop on all possibilities of mix dataset-split-scaling-algorithms
    for dataset in datasets:
        for seed in range(1,4): #1-2-3
            for norm in range(1,5):#1-2-3-4
                #switch(seed):
                x_train, x_test, y_train, y_test = pp.preprocessing(dataset[1], seed, norm)

                regressor_lr = lr.train(x_train, y_train)
                regressor_rt = rt.train(x_train,y_train)

                y_pred_lr = lr.predict(regressor_lr, x_test)
                y_pred_rt = rt.predict(regressor_rt,x_test)

                mae_lr = round(analyse.mae(y_test, y_pred_lr),2)
                mae_rt = round(analyse.mae(y_test,y_pred_rt),2)

                r2_lr = round(analyse.r2(y_test, y_pred_lr),2)
                r2_rt = round(analyse.r2(y_test, y_pred_rt),2)

                rmse_lr = round(analyse.rmse(y_test, y_pred_lr),2)
                rmse_rt = round(analyse.rmse(y_test, y_pred_rt),2)

                returnArray.append("Linear regression : dataset : " + dataset[0] + ", seed : " + str(seed) + ", norm : " + str(norm) +", mae : " + str(mae_lr) + ", r2 :" +
                                   str(r2_lr) + ", rmse : " + str(rmse_lr) + "\n")
                returnArray.append("Regression tree   : dataset : " + dataset[0] + ", seed : " + str(seed) + ", norm : " + str(norm) +", mae : " + str(mae_rt) + ", r2 :" +
                                   str(r2_rt) + ", rmse : " + str(rmse_rt) + "\n")
    return returnArray


def save_into_file(fileName, data):
    """
    Save values into a file to have a better view a check reproducibility of code

    :param fileName: Str

    :param data: array of string containing all values

    :return:
    """
    f = open(fileName,"w+")
    for i in data:
        f.write(i)
    f.close()

if __name__ == "__main__":
    all_possibilities = run_all_possibilities()
    print(len(all_possibilities))
    save_into_file("test.txt", all_possibilities)