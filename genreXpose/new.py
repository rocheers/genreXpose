import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

from utils import read_mfcc_file, genre_list


def kNN_model(X, Y, n):
    cv = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, random_state=0)

    train_errors = []
    test_errors = []

    scores = []

    clfs = []  # for the median

    cms = []
    labels = np.unique(Y)

    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

        clf = KNeighborsClassifier(n_neighbors=n, weights='distance')
        clf.fit(X_train, y_train)
        clfs.append(clf)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        scores.append(test_score)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cms.append(cm)
   
    return np.mean(train_errors), np.mean(test_errors), np.asarray(cms), clf


def svm_model(train_X, train_y):
    cv = ShuffleSplit(n=len(train_X), n_iter=10, test_size=0.3, random_state=0)

    train_errors = []
    test_errors = []

    scores = []

    labels = np.unique(train_y)

    min_score = 1e5

    for train, test in cv:
        X_train, y_train = train_X[train], train_y[train]
        X_test, y_test = train_X[test], train_y[test]

        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        scores.append(test_score)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        if test_score < min_score:
            min_score = test_score
            best_clf = clf

    return np.mean(train_errors), np.mean(test_errors), best_clf


def kmean_model(train_X, train_y, test_X, test_y):
    labels = np.unique(train_y)
    clt = KMeans(n_clusters=len(labels), max_iter=1000)
    kmeans = clt.fit(train_X)
    print(kmeans.labels_)


def run_model(model, train_X, train_y, test_X, test_y, save=True):
    if model == 'knn':
        y, clfs = [], []
        n = 41
        for i in range(1, n):
            train_error, test_error, cms, clf = kNN_model(train_X, train_y, i)
            y.append(test_error)
            clfs.append(clf)

        clf = clfs[y.index(min(y))]
        pred_y = clf.predict(test_X)
        cm = confusion_matrix(test_y, pred_y)
        test_error = 1 - clf.score(test_X, test_y)

    elif model == 'svm':
        train_error, test_error, clf = svm_model(train_X, train_y)
        pred_y = clf.predict(test_X)
        cm = confusion_matrix(test_y, pred_y)
        test_error = 1 - clf.score(test_X, test_y)

    print(test_error, cm, sep="\n")

    if save:
        joblib.dump(clf, 'saved_model/model.pkl')

    return test_error, cm


def nn_model(train_X, train_y, test_X, test_y):
    mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(10, 5), max_iter=2000,
                        learning_rate_init=0.001, solver='adam', tol=1e-5, random_state=1)

    mlp.fit(train_X, train_y)
    print("Training set score: %f" % mlp.score(train_X, train_y))
    print("Test set score: %f" % mlp.score(test_X, test_y))
    pred_y = mlp.predict(test_X)
    plot_confusion_matrix(confusion_matrix(test_y, pred_y))


def plot_confusion_matrix(cm, classes=genre_list, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_PCA(train_X):
    reduced_data = PCA(n_components=2).fit_transform(train_X)
    print(reduced_data.shape)
    kmeans = KMeans(n_clusters=4).fit(reduced_data)
    # print(kmeans.labels_)

    h = 2     # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the audio dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


if __name__ == '__main__':
    train_X, train_y = read_mfcc_file(test=False)
    test_X, test_y = read_mfcc_file(test=True)
    # print(train_X.shape, test_X.shape)

    # ----------
    # kNN Classification
    # ----------
    # error, cm = run_model('knn', train_X, train_y, test_X, test_y, save=False)
    # plot_confusion_matrix(cm, genre_list)

    # ----------
    # SVM Classification
    # ----------
    # error, cm = run_model('svm', train_X, train_y, test_X, test_y, save=False)
    # plot_confusion_matrix(cm, genre_list)

    # ----------
    # K-means Clustering
    # ----------
    # kmean_model(train_X, train_y, test_X, test_y)
    # print(train_y)

    # ----------
    # Neural Network Classification
    # ----------
    nn_model(train_X, train_y, test_X, test_y)
