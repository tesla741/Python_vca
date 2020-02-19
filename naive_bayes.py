import os
import re
import string
import math
import random
import csv
from nltk.stem import WordNetLemmatizer
import pickle
import functools

DATA_DIR = 'data'
target_names = ['comp', 'n_comp']

def get_data(DATA_DIR):
    data = []
    target = []
    comp_files = os.listdir(os.path.join(DATA_DIR, 'comp'))
    for comp_file in comp_files:
        with open(os.path.join(DATA_DIR, 'comp', comp_file), encoding="latin-1") as f:
            lst_p = f.readlines()
            data.extend(lst_p)
            target.extend([1] * len(lst_p))

    n_comp_files = os.listdir(os.path.join(DATA_DIR, 'n_comp'))
    for n_comp_file in n_comp_files:
        with open(os.path.join(DATA_DIR, 'n_comp', n_comp_file), encoding="latin-1") as f:
            lst_n = f.readlines()
            data.extend(lst_n)
            target.extend([0] * len(lst_n))

    return data, target

class ComplianceDetector(object):
    """Implementation of Naive Bayes for binary classification"""
    def clean(self, s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)

    def lemmatize(self, words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def tokenize(self, text):
        text = self.clean(text).lower()
        words = re.split("\W+", text)
        lemmas = self.lemmatize(words)
        return lemmas

    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            if(word == ''):
                continue
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts


    def fit(self, X, Y):
        self.num_messages = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.p_w_given_class = {}
        self.vocab = set()
        self.thresh = {'thresh': 0.5}
        n = len(X)
        self.num_messages['comp'] = sum(1 for label in Y if label == 1)
        self.num_messages['n_comp'] = sum(1 for label in Y if label == 0)
        self.log_class_priors['comp'] = math.log(self.num_messages['comp'] / n)
        self.log_class_priors['n_comp'] = math.log(self.num_messages['n_comp'] / n)
        self.word_counts['comp'] = {}
        self.word_counts['n_comp'] = {}
        self.p_w_given_class['comp'] = {}
        self.p_w_given_class['n_comp'] = {}

        for x, y in zip(X, Y):
            c = 'comp' if y == 1 else 'n_comp'
            counts = self.get_word_counts(self.tokenize(x))
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0

                self.word_counts[c][word] += count

        for word in self.vocab:
            self.p_w_given_class['comp'][word] = math.log(
                (self.word_counts['comp'].get(word, 0.0) + 1) / (
                            sum(self.word_counts['comp'].values()) + len(self.vocab)))
            self.p_w_given_class['n_comp'][word] = math.log(
                (self.word_counts['n_comp'].get(word, 0.0) + 1) / (
                            sum(self.word_counts['n_comp'].values()) + len(self.vocab)))

        self.p_w_given_class['comp']['UNKNOWN'] = math.log(
            (0.0 + 1) / (
                    sum(self.word_counts['comp'].values()) + len(self.vocab)))

        self.p_w_given_class['n_comp']['UNKNOWN'] = math.log(
            (0.0 + 1) / (
                    sum(self.word_counts['n_comp'].values()) + len(self.vocab)))

        prob_pickle = open("probs.pickle", "wb")
        pickle.dump(self.p_w_given_class, prob_pickle)
        prob_pickle.close()

        prior_pickle = open("prior.pickle", "wb")
        pickle.dump(self.log_class_priors, prior_pickle)
        prior_pickle.close()

        vocab_pickle = open("vocab.pickle", "wb")
        pickle.dump(self.vocab, vocab_pickle)
        vocab_pickle.close()

        thresh_pickle = open("threshold.pickle", "wb")
        pickle.dump(self.thresh, thresh_pickle)
        thresh_pickle.close()

    def predict(self, X):
        result = []
        p_comp = []

        for x in X:
            counts = self.get_word_counts(self.tokenize(x))
            comp_score = 0
            n_comp_score = 0
            for word, _ in counts.items():
                if word in self.vocab:
                    log_w_given_comp = self.p_w_given_class['comp'][word]
                    log_w_given_n_comp = self.p_w_given_class['n_comp'][word]
                else:
                    log_w_given_comp = self.p_w_given_class['comp']['UNKNOWN']
                    log_w_given_n_comp = self.p_w_given_class['n_comp']['UNKNOWN']

                comp_score += log_w_given_comp
                n_comp_score += log_w_given_n_comp

            comp_score += self.log_class_priors['comp']
            n_comp_score += self.log_class_priors['n_comp']
            p1 = math.exp(comp_score)
            p2 = math.exp(n_comp_score)

            #print("Test Sentence: " + x + "\n")

            pcomp = float(p1)/(p1 + p2)

            #print("Compliance Score: "+ str(float(p1)/(p1 + p2)))

            p_comp.append(pcomp)

            if comp_score > n_comp_score:
                result.append(1)
            else:
                result.append(0)

        return result, p_comp

if __name__ == '__main__':

    X, y = get_data(DATA_DIR)

    combined = list(zip(X, y))
    random.shuffle(combined)

    X[:], y[:] = zip(*combined)

    MNB = ComplianceDetector()
    MNB.fit(X, y)

    pred, comp_s = MNB.predict(X)

    if functools.reduce(lambda i, j: i and j, map(lambda m, k: m == k, pred, y), True):
        print("The lists are identical")
    else:
        print("The lists are not identical")