import os
import re
MIN_FREQ = 3
def invert_dict(d):
    res = {}
    for k, v in d.iteritems():
        res[v] = k
    return res

def read_conll_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1],tokens[3]))
    return sents

def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1

def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab

def replace_word(word):
    """
        Replaces rare words with categories (numbers, dates, etc...)
    """
    ### YOUR CODE HERE
    twodigit = re.compile("[0-9][0-9]") # Two-digit number
    if bool(twodigit.match(word)):
        return "twoDigitNum"
    fourdigit = re.compile("[0-9],?[0-9][0-9][0-9]") # Four-digit number
    if bool(fourdigit.match(word)):
        return "fourDigitNum"
    if bool(re.search('[0-9]', word)) and bool(re.search('[a-zA-Z]', word)): # contains alpha and digit
        return "containsDigitAndAlpha"
    digitanddash = re.compile("\d") # contains dash and digit
    if bool(digitanddash.match(word)) and '-' in word:
        return "containsDigitAndDash"
    digitandslash = re.compile("\d") # contains slash and digit
    if bool(digitandslash.match(word)) and '/' in word:
        return "containsDigitAndSlash"
    digitandcomma = re.compile("\d") # contains comma and digit
    if bool(digitandcomma.match(word)) and ',' in word:
        return "containsDigitAndComma"
    digitandperiod = re.compile("\d") # contains period and digit
    if bool(digitandperiod.match(word)) and '.' in word:
        return "containsDigitAndPeriod"
    company1 = re.compile("^[A-Z]+[a-z]+-[A-Z]+[a-z]+$")  # company word
    company2 = re.compile("([A-Z].[A-Z])+")
    if bool(company1.match(word)) or bool(company2.match(word)):
        return "company"
    if word.endswith('ly'): # adverb
        return "adverb"
    person = re.compile("^[A-Z][a-z]+ [A-Z][a-z]+$")  # person
    if bool(person.match(word)):
        return "person"
    if word.endswith('ing'):  # ing
        return "ing"
    if re.search(r'(fy)',word):
        return '_verblike_'
    if word in {'what', 'how', 'when', 'who', 'whom', 'where', 'which', 'whose', 'why'}:
        return "w_h"
    if re.search(r'(ion|ty|ics|ment|ence|ance|ness|ist|ism|acy|ice|sion|tion|ency|esis|osis|cian|hood|logy|dom|age)', word):
        return '_NOUNLIKE_'
    if re.search(r'(ish|able|ables|ful|less)', word):
        return '_ADJLIKE_'
    if word.endswith('est'):
        return 'superlative'
    if word.endswith('er') or word == 'less' or word == 'more' or word == 'worse':
        return 'comparative'
    if word in {'this', 'that', 'these', 'those','such', 'rather', 'quite'}:
        return "determines"
    if word.lower() in {'sunday','monday','tuesday','wednesday','thursday','friday','saturday','yesterday'}:
        return "day"
    if word in {'i', 'he', 'she', 'you', 'it'}:
        return 'personal_pronoun'
    if word in {'but', 'nor', 'or', 'yet', 'so'}:
        return 'Coordinating_Conjunction'
    capperiod = re.compile("^[A-Z].$")  # cap period
    if bool(capperiod.match(word)):
        return "capPeriod"
    if word.isupper():
        return "allCaps"
    capword = re.compile("^[A-Z][a-z]+$")  # cap word
    if bool(capword.match(word)):
        return "initCap"
    if word.islower():
        print word
        return "lowerCase"
    ### END YOUR CODE
    return "UNK"

def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                new_sent.append((replace_word(token[0]), token[1]))
                replaced += 1
            total += 1
        res.append(new_sent)
    print "replaced: " + str(float(replaced)/total)
    return res







