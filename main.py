import numpy as np
import csv

def argmax(args,f):
    maxArg = None
    maxVal = 0
    for arg in args:
        val = f(*arg)
        if val > maxVal:
            maxVal = val
            maxArg = arg
    return maxArg

class SMR: #Summary Struct
    def __init__(self,f):
        f = list(f)
        #print(f)
        self.mean = np.mean(f)
        self.stddev = np.std(f)
    def __str__(self):
        return "mean : {}, stddev : {}".format(self.mean,self.stddev)
    def __repr__(self):
        return self.__str__()

class GNB: #Gaussian Naive Bayes
    def __init__(self):
        self.dat = {}
        self.summary = {}

    def Prob(self,x,m,d):
        return 1.0 / np.sqrt(2*np.pi) * np.exp(-(x-m)**2/(2*d**2))

    def fit(self,xs,ys):
        self.num_features = len(xs)
        #classify by labels
        for x,y in zip(xs,ys):
            #print(y)
            try:
                self.dat[y] += [x]
            except:
                self.dat[y] = [x]
        #print self.dat
        #summarize each label
        self.summarize()

    def summarize(self):
        self.summary = {} #[[] for _ in range(len(self.dat))]
        for lab,insts in self.dat.items(): #label, instances
            fs = zip(*insts) #features
            self.summary[lab] = [[] for _ in range(len(fs))]
            for fi, f in enumerate(fs): #feature
                self.summary[lab][fi] = SMR(f)
        print self.summary

    def predict(self,xs):
        maxP = 0.0
        maxY = None
        totP = 0.0
        for y, s in self.summary.items():
            p = 1.0
            for i,x in enumerate(xs):
                p *= self.Prob(x,s[i].mean,s[i].stddev)
            if p > maxP:
                maxP = p
                maxY = y
            totP += p
        return maxY,(maxP/totP)

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
    dataset = readData("pima-indians-diabetes.data.csv")
    x_train,y_train,x_test,y_test = splitData(dataset,0.67)

    gnb = GNB()
    gnb.fit(x_train,y_train)

    correct = 0

    for x,t in zip(x_test,y_test):#t = target
        y,p = gnb.predict(x)
        if y==t:
            correct += 1
        print "Y : {}, T : {}, P : {:2.2f}".format(y,t,p)

    print "ACCURACY : {:2.2f}%".format(correct * 100.0/len(y_test))

if __name__ == "__main__":
    main()
