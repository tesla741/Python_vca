import re, string, unicodedata
import nltk
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas

def prepare_dataset(file_name):
    lines = open(file_name, "r").readlines()
    comp = []
    n_comp = []
    flag = 0
    for idx, line in enumerate(lines):
        a,b,c,d,e,f,g = '','','','','','',''
        if("]" in line):
            a, b = line.strip().split("[")
            c, d = b.strip().split("]")
            e, f, g = c.strip().split(", ")
        if(idx%2 == 0):
            if(a != ''):
                comp.append(a.strip() + " " + e.strip() + " " + d.strip())
                comp.append(a.strip() + " " + f.strip() + " " + d.strip())
                comp.append(a.strip() + " " + g.strip() + " " + d.strip())
                flag = 1
            else:
                comp.append(line)
        else:
            if(a != ''):
                n_comp.append(a.strip() + " " + e.strip() + " " + d.strip())
                n_comp.append(a.strip() + " " + f.strip() + " " + d.strip())
                n_comp.append(a.strip() + " " + g.strip() + " " + d.strip())
                flag = 0
            else:
                if(flag == 1):
                    n_comp.append(line)
                    n_comp.append(line)
                    n_comp.append(line)
                    flag = 0
                else:
                    n_comp.append(line)
    return comp, n_comp

def write_prepared_dataset(list, filename):
    f = open(filename, "w")
    for var in list:
        f.write(var.strip() + "\n")
    f.close()

def processed_lines(list):
    ret_lst = []
    for sample in list:
        sample = denoise_text(sample)
        words = nltk.word_tokenize(sample)
        words = normalize(words)
        lemmas = lemmatize(words)
        ret_lst.append(lemmas)
    return ret_lst

d = {'please':10, 'hold': 10, 'stay':10, 'disconnect':10, 'call':10, 'back':10}
def weight_count(list):
    t_w = 0
    for var in list:
        if(var in d.keys()):
           t_w += d[var]
        else:
            t_w += 1
    return t_w

def weighted_jaccard_sim(list1, list2):
    print("Ref: ")
    print(sorted(list2))
    print("test: ")
    print(sorted(list1))
    inter = list(set(list1).intersection(list2))
    uni = list(set(list1).union(list2))
    print("Intersection: ")
    print(sorted(inter))
    uni = list(set(list1).union(list2))
    print("Union: ")
    print(sorted(uni))
    total_weight_inter = weight_count(inter)
    total_weight_uni = weight_count(uni)
    print("intersection_card: " + str(total_weight_inter))
    print("union_card: " + str(total_weight_uni))
    print("JS_weighted: " + str(float(total_weight_inter)/total_weight_uni))
    return float(total_weight_inter)/total_weight_uni

def jaccard_similarity(list1, list2):
    print("Ref: ")
    print(sorted(list2))
    print("test: ")
    print(sorted(list1))
    inter = list(set(list1).intersection(list2))
    inter_card = len(inter)
    print("Intersection: ")
    print(sorted(inter))
    print("intersection_card: " + str(inter_card))
    uni = list(set(list1).union(list2))
    print("Union: ")
    print(sorted(uni))
    uni_card = len(uni)
    print("union_card: " + str(uni_card))
    print("JS: " + str(float(inter_card) / uni_card))
    return float(inter_card) / uni_card

def calculate_similarity_all_examples(lst, x_0, filename):
    f = open(filename, "w")
    max = 0
    min = 999999
    for indx, list1 in enumerate(lst):
        score = weighted_jaccard_sim(list1, x_0)
        if (score > max):
            max = score
        if (score < min):
            min = score
        f.write(lst[indx].strip() + "," + str(score) + "\n")
    f.write("\n\n-------Range of scores: " + str(min) + " - " + str(max) + " ---------------------------------------")

data = "dataset"
comp, n_comp = prepare_dataset(data)
write_prepared_dataset(comp, "comp.txt")
write_prepared_dataset(n_comp, "n_comp.txt")