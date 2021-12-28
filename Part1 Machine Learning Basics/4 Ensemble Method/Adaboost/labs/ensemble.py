import pickle

import numpy as np


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.n_weakers = n_weakers_limit
        self.clf = [weak_classifier(max_depth=8, min_samples_split=10) for i in range(n_weakers_limit)]
        self.clf_weights = np.zeros(n_weakers_limit, dtype=float)

    def is_good_enough(self, ac):
        '''Optional'''
        if (1 - ac) < 1e-3:
            print('stop now')
            return True


    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        weights = 1 / np.ones(np.shape(X)[0])
        for i in range(self.n_weakers):
            self.clf[i].fit(X, y, sample_weight=weights)

            acc = self.clf[i].score(X, y)
            print("Week Classifier %d accuracy is : %f" % (i, acc))
            alpha = np.log(acc / (1 - acc + 1e-3)) / 2
            self.clf_weights[i] = alpha

            if self.is_good_enough(acc) == True:
                break

            predict = self.clf[i].predict(X)
            sum_weight = np.sum(weights)
            weights[predict == y] *= np.exp(-alpha) / sum_weight
            weights[predict != y] *= np.exp(alpha) / sum_weight

        print(self.clf_weights)



    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        scores = None
        for i in range(self.n_weakers):
            if self.clf_weights[i] > 1e-3:
                result = self.clf[i].predict(X).astype(float)
                result *= self.clf_weights[i]
                if scores is None:
                    scores = result
                else:
                    scores += result
        return scores


    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        scores = self.predict_scores(X)
        scores /= self.n_weakers
        labels = np.where(0.5 < scores, 1, 0)
        return labels

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
