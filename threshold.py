import os
import re
import string
import math
import random
import csv
from nltk.stem import WordNetLemmatizer
import pickle
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

DATA_DIR = 'data'
target_names = ['comp', 'n_comp']

def get_data(DATA_DIR):
    data_p = []
    target_p = []
    data_n = []
    target_n = []
    comp_files = os.listdir(os.path.join(DATA_DIR, 'comp'))
    for comp_file in comp_files:
        with open(os.path.join(DATA_DIR, 'comp', comp_file), encoding="latin-1") as f:
            lst_p = f.readlines()
            data_p.extend(lst_p)
            target_p.extend([1] * len(lst_p))

    n_comp_files = os.listdir(os.path.join(DATA_DIR, 'n_comp'))
    for n_comp_file in n_comp_files:
        with open(os.path.join(DATA_DIR, 'n_comp', n_comp_file), encoding="latin-1") as f:
            lst_n = f.readlines()
            data_n.extend(lst_n)
            target_n.extend([0] * len(lst_n))

    return data_p, target_p, data_n, target_n


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
            if (word == ''):
                continue
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    def fit(self, X, Y):
        self.num_messages = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.p_w_given_class = {}
        self.vocab = set()

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

        prob_pickle = open("probs.pickle", "wb")
        pickle.dump(self.p_w_given_class, prob_pickle)
        prob_pickle.close()

        prior_pickle = open("prior.pickle", "wb")
        pickle.dump(self.log_class_priors, prior_pickle)
        prior_pickle.close()

        vocab_pickle = open("vocab.pickle", "wb")
        pickle.dump(self.vocab, vocab_pickle)
        vocab_pickle.close()

    def predict(self, X):
        result = []
        p_comp = []

        for x in X:
            counts = self.get_word_counts(self.tokenize(x))
            comp_score = 0
            n_comp_score = 0
            for word, _ in counts.items():
                if word not in self.vocab: continue

                # add Laplace smoothing
                log_w_given_comp = math.log(
                    (self.word_counts['comp'].get(word, 0.0) + 1) / (
                                sum(self.word_counts['comp'].values()) + len(self.vocab)))

                log_w_given_n_comp = math.log(
                    (self.word_counts['n_comp'].get(word, 0.0) + 1) / (
                                sum(self.word_counts['n_comp'].values()) + len(self.vocab)))

                # print(word + " probability of word given compliance: " + str(math.exp(log_w_given_comp)))
                # print(word + " probability of word given non-compliance: " + str(math.exp(log_w_given_n_comp)))

                comp_score += log_w_given_comp
                n_comp_score += log_w_given_n_comp

            comp_score += self.log_class_priors['comp']
            n_comp_score += self.log_class_priors['n_comp']
            p1 = math.exp(comp_score)
            p2 = math.exp(n_comp_score)

            # print("Test Sentence: " + x + "\n")

            pcomp = float(p1) / (p1 + p2)

            # print("Compliance Score: "+ str(float(p1)/(p1 + p2)))

            p_comp.append(pcomp)

            if comp_score > n_comp_score:
                result.append(1)
            else:
                result.append(0)

        return result, p_comp


if __name__ == '__main__':
    X_p, y_p, X_n, y_n = get_data(DATA_DIR)

    flag = 30
    while (flag):
        X_test = []
        y_test = []

        combined = list(zip(X_p, y_p))
        random.shuffle(combined)

        X_p[:], y_p[:] = zip(*combined)


        combined = list(zip(X_n, y_n))
        random.shuffle(combined)

        X_n[:], y_n[:] = zip(*combined)

        X_train_p = X_p[:21]
        X_train_n = X_n[:21]
        y_train_p = y_p[:21]
        y_train_n = y_n[:21]

        X_test_p = X_p[21:]
        X_test_n = X_n[21:]
        y_test_p = y_p[21:]
        y_test_n = y_n[21:]

        X_test.extend(X_train_p)
        X_test.extend(X_train_n)
        y_test.extend(y_train_p)
        y_test.extend(y_train_n)

        X_train = []
        y_train = []

        X_train.extend(X_train_p)
        X_train.extend(X_train_n)
        y_train.extend(y_train_p)
        y_train.extend(y_train_n)

        MNB = ComplianceDetector()
        #count_vect = CountVectorizer()
        #X_train_counts = count_vect.fit_transform(X_train)
        MNB.fit(X_train, y_train)
        #clf = MultinomialNB()
        #clf.fit(X_train_counts, y_train)

        # ref = open("./reference", "r").readlines()
        # ref_counts = count_vect.transform(ref)
        # pred = clf.predict(ref_counts)
        # comp_s = clf.predict_proba(ref_counts)[:,1]
        pred, comp_s = MNB.predict(X_test)

        pos_list_index = [i for i in range(len(y_test)) if y_test[i] == 1]

        pos_list_prob = list(itemgetter(*pos_list_index)(comp_s))

        #pos_list_sent = list(itemgetter(*pos_list_index)(X_test))

        neg_list_index = [i for i in range(len(y_test)) if y_test[i] == 0]

        neg_list_prob = list(itemgetter(*neg_list_index)(comp_s))

        #neg_list_sent = list(itemgetter(*neg_list_index)(X_train))

        # res = open("./results_comp.csv", "w", newline='')
        # writer = csv.writer(res)
        # writer.writerow(["Sentence", "Compliance probability"])
        # for var1, var2 in zip(pos_list_sent, pos_list_prob):
        #     writer.writerow([var1, var2])
        # res.close()
        #
        # res = open("./results_n_comp.csv", "w", newline='')
        # writer = csv.writer(res)
        # writer.writerow(["Sentence", "Compliance probability"])
        # for var1, var2 in zip(neg_list_sent, neg_list_prob):
        #     writer.writerow([var1, var2])
        # res.close()
        thresh = min(pos_list_prob) + max(neg_list_prob)
        print("Threshold: ", thresh/2)

        flag -= 1