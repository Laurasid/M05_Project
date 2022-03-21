from preprocessor import preprocessing as pp
from model import linear_regression as lr
from model import regression_tree as rt
from analysis import analyse


def run_all_possibilities():
    returnArray = []
    datasets = []

    dataset1 = pp.import_dataset("Data/winequality-red.csv")
    dataset2 = pp.import_dataset("Data/winequality-white.csv")
    dataset3 = pp.import_dataset("Data/housing.data")

    datasets.append(("red_Wine",dataset1))
    datasets.append(("white_wine",dataset2))
    datasets.append(("housing",dataset3))

    for dataset in datasets:
        for seed in range(1,4): #1-2-3
            for norm in range(1,5):
                #switch(seed):
                x_train, x_test, y_train, y_test = pp.preprocessing(dataset[1], seed, norm)

                regressor_lr = lr.train(x_train, y_train)
                regressor_rt = rt.train(x_train,y_train)

                y_pred_lr = lr.predict(regressor_lr, x_test)
                y_pred_rt = rt.predict(regressor_rt,x_test)

                mae_lr = analyse.mae(y_test, y_pred_lr)
                mae_rt = analyse.mae(y_test,y_pred_rt)

                r2_lr = analyse.r2(y_test, y_pred_lr)
                r2_rt = analyse.r2(y_test, y_pred_rt)

                rmse_lr = analyse.rmse(y_test, y_pred_lr)
                rmse_rt = analyse.rmse(y_test, y_pred_rt)

                returnArray.append("Linear regression : dataset : " + dataset[0] + ", seed : " + str(seed) + ", norm : " + str(norm) +", mae : " + str(mae_lr) + ", r2 :" +
                                   str(r2_lr) + ", rmse : " + str(rmse_lr) + "\n")
                returnArray.append("Regression tree   : dataset : " + dataset[0] + ", seed : " + str(seed) + ", norm : " + str(norm) +", mae : " + str(mae_rt) + ", r2 :" +
                                   str(r2_rt) + ", rmse : " + str(rmse_rt) + "\n")


    return returnArray

def save_into_file(fileName, data):
    f = open(fileName,"w+")
    for i in data:
        f.write(i)
    f.close()

if __name__ == "__main__":
    all_possibilities = run_all_possibilities()
    print(len(all_possibilities))
    save_into_file("test.txt", all_possibilities)