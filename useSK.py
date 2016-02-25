from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import csv

def readData(fileName):
    lines = csv.reader(open(fileName,"rb"))
    return list(lines) 

def splitData(dataset,pivot):
    pivot = int(round(pivot * len(dataset)))
    np.random.shuffle(dataset)
    xs = [map(float,x[:-1]) for x in dataset]
    ys = [int(x[-1]) for x in dataset]
    x_train = xs[:pivot]
    y_train = ys[:pivot]
    x_test = xs[pivot:]
    y_test = ys[pivot:]
    return x_train,y_train,x_test,y_test

def main():
    clf = GaussianNB()
    dataset = readData("pima-indians-diabetes.data.csv")
    pivot = 0.67
    x_train,y_train,x_test,y_test = splitData(dataset,pivot)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print "ACCURACY : {:2.2f}%".format(100.0 * accuracy_score(y_test,y_pred))


if __name__ == "__main__":
    main()
