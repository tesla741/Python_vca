import re
import string
import math
from nltk.stem import WordNetLemmatizer
import pickle
import json
from cTopic import cTopic

class ComplianceDetector(object):
    """Implementation of Compliance Detector"""
    def __init__(self):
        self.vocab = pickle.load(open("vocab.pickle","rb"))
        self.log_class_priors = pickle.load(open("prior.pickle","rb"))
        self.p_w_given_class = pickle.load(open("probs.pickle","rb"))
        conf = json.loads(open("config.json", "r").read())
        self.threshold = conf["thresh_nb"]
        self.min = conf["min"]
        self.max = conf["max"]

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

    def predict(self, x):

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

        pcomp = float(p1) / (p1 + p2)

        if pcomp > self.threshold:
            return True
        else:
            return False

    def count_words(self, para):
        words = para.strip().split()
        return words, len(words)

def process(para, cmp_det):
    worthy_chunks_c = {}
    worthy_chunks_nc = {}
    minimum  = cmp_det.min
    max = cmp_det.max
    yy = {}
    yN = {}
    words_para, N = cmp_det.count_words(para)
    cnt_deno = 0
    cnt_ch = 0
    for i in range(0, N-minimum+1):
        check_big_c = False
        check_big_nc = False
        cnt_ch += len(range(minimum, min(max+1, N - i + 1)))
        for k in reversed(range(minimum,min(max+1, N - i + 1))):
            j = i + k
            ch_words = words_para[i:j+1]
            chunk = " ".join(ch_words)
            #print(chunk)
            f1 = cTopic(cmp_det, chunk)
            if(not f1):
                break
            f2 = cmp_det.predict(chunk)
            if(f1 and f2):
                if(not check_big_c):
                    check_big_c = True
                    worthy_chunks_c[i] = j
                    if(i in yy.keys()):
                        yy[i].append(j)
                    else:
                        yy[i] = [j]
                cnt_deno += 1
            elif(f1 and not f2):
                if(not check_big_nc):
                    check_big_nc = True
                    worthy_chunks_nc[i] = j
                if(i in yN.keys()):
                    yN[i].append(j)
                else:
                    yN[i] = [j]
                cnt_deno += 1
    return cnt_ch, cnt_deno, yy, yN, worthy_chunks_c, worthy_chunks_nc

def isCovered(subset_key, subset_value, superset):
    superset_keys = superset.keys()
    for key in superset_keys:
        if subset_key >= key:
            superset_values = superset[key]
            for val in superset_values:
                if subset_value <= val:
                    return True
    return False

def infer_compliance(yy, yn, denominator):
    numerator = 0
    if denominator == 0:
        return
    for key in yn.keys():
        for val in yn[key]:
            if (not isCovered(key, val, yy)):
                numerator += 1
    return numerator

def compliance_score(text):
    cmp_det = ComplianceDetector()
    threshold = 0.6
    cnt_ch, cnt_deno, yy, yN, worthy_chunks_c, worthy_chunks_nc = process(text, cmp_det)
    cnt_unc = infer_compliance(yy, yN, cnt_deno)
    all_words = text.strip().split()
    final_chunk = ''
    if(cnt_ch == 0):
        f1 = cTopic(cmp_det, text)
        f2 = cmp_det.predict(text)
    else:
        f1 = float(cnt_deno)/ cnt_ch
        if(f1 > 0 ):
            f2 = float(cnt_unc) / cnt_deno
        else:
            f2 = 'nan'

    if(f2 == 'nan' or f2 <= threshold):
        flag  = "Yes"
        if(len(worthy_chunks_c) > 0):
            mini  = min(worthy_chunks_c.keys())
            s_mini = mini
            for var in worthy_chunks_c.keys():
                if(var < worthy_chunks_c[mini]):
                    if (worthy_chunks_c[var] > worthy_chunks_c[mini]):
                        mini = var
            final_chunk = " ".join(all_words[s_mini: worthy_chunks_c[mini] + 1])
        else:
            final_chunk = "EMPTY"
    else:
        flag = "No"
        if(len(worthy_chunks_nc) > 0):
            mini = min(worthy_chunks_nc.keys())
            s_mini = mini
            for var in worthy_chunks_nc.keys():
                if(var <= worthy_chunks_nc[mini]):
                    if(worthy_chunks_nc[var] > worthy_chunks_nc[mini]):
                        mini = var
            final_chunk = " ".join(all_words[s_mini: worthy_chunks_nc[mini] + 1])
        else:
            final_chunk = "EMPTY"
    if(f2 != 'nan'):
        # print(flag + ": <" + '%.3f'%(f1) + "," '%.3f'%(1 - f2) + ">")
        formatted_prob=(round(1-f2, 2)) 
        print(flag,formatted_prob,final_chunk)
        return flag, formatted_prob, final_chunk   
    else:
        # print(flag + ": <" + '%.3f' % (f1) + ",", "nan" + ">")
        return "NW", "NS", final_chunk
        print("NW","NS",final_chunk)

    # print("FINAL_CHUNK: ", final_chunk)

# trans = "I will place your call on hold while I call the [hotel, airline, other vendor]. In case we get disconnected, I will call you back on the number you are calling us from."
# compliance_score(trans)

