import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import hmms
from sklearn.metrics import accuracy_score
import json
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def cluster_data(data):
    index = len(data) - 1
    X = list(data['Transaction_Value'])
    X_2di = list()
    for i in X:
        X_2di.append([i, 0])
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_2di[:index])
    test = kmeans.predict(X_2di[index:])
    max_value = max(X[:index])
    max_value_index = X.index(max_value)
    min_value = min(X[:index])
    min_value_index = X.index(min_value)
    obver = [0] * len(X)
    A = np.hstack([kmeans.labels_, test])
    ma = A[max_value_index]
    mi = A[min_value_index]
    for i in range(len(X)):
        if A[i] == ma:
            obver[i] = 2
        elif A[i] == mi:
            obver[i] = 0
        else:
            obver[i] = 1
    return obver


def train_model(data, epoch, eps):
    A = np.array([[1 / 4, 1 / 4, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                  [1 / 4, 1 / 4, 1 / 4, 1 / 4]])
    B = np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
    Pi = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])

    index = len(data) - 1
    train = np.array([data[:index]])
    dhmm = hmms.DtHMM(A, B, Pi)
    dhmm.baum_welch(train, epoch)
    base = np.array([data[:index]])
    test = np.array([data[1:1 + index]])
    result = (np.exp(dhmm.data_estimate(base)) - np.exp(dhmm.data_estimate(test))) / np.exp(dhmm.data_estimate(base))
    if result < eps:
        flag = 0
    else:
        flag = 1
    return flag


def testing(eps):
    list_predict = list()
    for i in range(470):
        path_file = './data2/TransactionID_' + str(i + 1) + '.csv'
        data = pd.read_csv(path_file)
        temp = cluster_data(data)
        list_predict.append(train_model(temp, 10, eps))
    with open('./raw/dict_fraud_transaction.json') as f:
        Check = json.load(f)
    B = list(Check.keys())
    A = [0] * 470
    for i in B:
        A[int(i) - 1] = 1
    # print("Accuracy: {}".format(accuracy_score(A, list_predict)))
    # print("Precision score: {}".format(precision_score(A, list_predict)))
    # print("Recall score: {}".format(recall_score(A, list_predict)))
    # print("Confusion matrix: ", confusion_matrix(A, list_predict))
    return accuracy_score(A, list_predict), precision_score(A, list_predict), recall_score(A, list_predict)


def main():
    x = np.arange(0.1, 0.6, 0.01)
    list_acc = list()
    list_pre = list()
    list_re = list()
    for i in x:
        t1, t2, t3 = testing(i)
        list_acc.append(t1)
        list_pre.append(t2)
        list_re.append(t3)
    plt.plot(x, list_acc, 'r-', label='accuracy')
    plt.plot(x, list_pre, 'b-', label='precision')
    plt.plot(x, list_re, 'g-', label='recall')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
