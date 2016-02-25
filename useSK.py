from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import csv
from matplotlib import pyplot as plt
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
    scores = []
    num_iter = 100
    for i in range(num_iter):
        x_train,y_train,x_test,y_test = splitData(dataset,pivot)
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        score = accuracy_score(y_test,y_pred)
        print "ACCURACY : {:2.2f}%".format(100.0 * score)
        scores += [score]

    plt.plot(scores)
    plt.title("Accuracy over {} iterations".format(num_iter))
    plt.show()



if __name__ == "__main__":
    main()
