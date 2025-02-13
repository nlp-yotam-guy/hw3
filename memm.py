from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import numpy as np


def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """

    extra_decoding_arguments = {}
    ### YOUR CODE HERE
    extra_decoding_arguments['S'] = dict()
    for sent in train_sents:
        for j in range(len(sent)):
            word = sent[j][0]
            tag = sent[j][1]
            if word not in extra_decoding_arguments['S']:
                extra_decoding_arguments['S'][word] = set()
            extra_decoding_arguments['S'][word].add(tag)
    ### END YOUR CODE

    return extra_decoding_arguments


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    ### YOUR CODE HERE
    features['prev_word'] = prev_word
    features['prev_prev_word'] = prevprev_word
    features['prev_label'] = prev_tag
    features['prev_prev_label'] = prevprev_tag
    features['next_word'] = next_word
    for i in xrange(1, min(5, len(curr_word))):
        features['prefix' + str(i)] = curr_word[:i]
    for i in xrange(1, min(5, len(curr_word))):
        features['suffix' + str(i)] = curr_word[-i:]
    if any(x.isdigit() for x in curr_word):
        features['has_number'] = 1
    features['is_lower'] = curr_word.islower()
    features['is_upper'] = curr_word.isupper()
    features['length'] = len(curr_word)
    if '-' in curr_word:
        features['contains_hyphen'] = 1
    features['prev_prev_tag'] = str(prev_tag) + '^' + str(prevprev_tag)
    features['prev_word_tag'] = str(prev_word) + '^' + str(prev_tag)
    features['prev_prev_word_tag'] = str(prevprev_word) + '^' + str(prevprev_tag)
    ### END YOUR CODE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))

    ### YOUR CODE HERE
    for i in range(len(sent)):
        # get features dictionary of word i'th in sentence
        features = extract_features(sent,i)
        # vectorize dictionary to use in the model
        v = vectorize_features(vec,features)
        # predict tag index (returns numpy array of length 1)
        pred = logreg.predict(v)
        # turn index to tag using the dictionary
        predicted_tags[i] = index_to_tag_dict[pred[0]]

    ### END YOUR CODE
    return predicted_tags

def calc_q(sent,vec,k,u,v,t):
    featurs = extract_features(sent,k)
    featurs['prev_tag'] = u
    featurs['prev_prev_tag'] = t
    featurs['word'] = sent[k][0]
    v_features = vectorize_features(vec,featurs)
    pred = logreg.predict_proba(v_features).flatten()
    idx = tag_to_idx_dict[v]
    return pred[idx]


def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    # init dictionaries for memoization
    pi = dict()
    bp = dict()
    S = dict()
    n = len(sent)
    S[-1] = {'*'}
    S[-2] = {'*'}
    # base case
    pi[(-1,'*','*')] = 1.0
    max_prob = float('-inf')
    for k in range(n):
        features = extract_features(sent, k)
        vectorized = vectorize_features(vec, features)
        # use predict to get the indices of all possible tags
        tags_idx = logreg.predict_proba(vectorized).flatten()
        # convert all indices to tags
        S[k] = extra_decoding_arguments['S'][sent[k][0]]
        # rest of viterbi algorithm
        end_max_prob = float('-Inf')
        for u in S[k-1]:
            for v in S[k]:
                max_prob = float('-inf')
                # bp_max = None
                for t in S[k-2]:
                    q = calc_q(sent,vec,k,u,v,t)
                    prob = pi[(k-1, t, u)] * q
                    if prob > max_prob:
                        max_prob = prob
                        pi[(k,u,v)] = prob
                        bp[(k,u,v)] = t

            if k == n-1 and pi[(k,u,v)] > end_max_prob:
                end_max_prob = pi[(k,u,v)]
                yn = v
                ym = u
    predicted_tags[n-1] = yn
    predicted_tags[n-2] = ym
    for j in range(n-3,-1,-1):
        predicted_tags[j] = bp[(j+2,predicted_tags[j+1],predicted_tags[j+2])]

    ### END YOUR CODE
    return predicted_tags

def should_log(sentence_index):
    if sentence_index > 0 and sentence_index % 10 == 0:
        if sentence_index < 150 or sentence_index % 200 == 0:
            return True

    return False


def memm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    correct_greedy_preds = 0
    correct_viterbi_preds = 0
    total_words_count = 0

    for i, sen in enumerate(test_data):
        ### YOUR CODE HERE
        # get predictions
        greedy_preds = memm_greedy(sen,logreg,vec,index_to_tag_dict,extra_decoding_arguments)
        viterbi_preds = memm_viterbi(sen,logreg,vec,index_to_tag_dict,extra_decoding_arguments)
        # calculate sentence accuracy
        for j in range(len(sen)):
            total_words_count+=1
            if sen[j][1] == greedy_preds[j]:
                correct_greedy_preds += 1
            if sen[j][1] == viterbi_preds[j]:
                correct_viterbi_preds += 1
            acc_greedy = float(correct_greedy_preds) / float(total_words_count)
            acc_viterbi = float(correct_viterbi_preds) / float(total_words_count)
        ### END YOUR CODE

        if should_log(i):
            if acc_greedy == 0 and acc_viterbi == 0:
                raise NotImplementedError
            eval_end_timer = time.time()
            print str.format("Sentence index: {} greedy_acc: {}    Viterbi_acc:{} , elapsed: {} ", str(i), str(acc_greedy), str(acc_viterbi) , str (eval_end_timer - eval_start_timer))
            eval_start_timer = time.time()

    acc_greedy = float(correct_greedy_preds) / float(total_words_count)
    acc_viterbi = float(correct_viterbi_preds) / float(total_words_count)

    return str(acc_viterbi), str(acc_greedy)

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict

# ****** remember to remove data slicing *********
if __name__ == "__main__":
    full_flow_start = time.time()
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)


    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    #sliced here
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print "Fitting..."
    start = time.time()
    #sliced here
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print "End training, elapsed " + str(end - start) + " seconds"
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print "Start evaluation on dev set"

    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
    end = time.time()
    print "Dev: Accuracy greedy memm : " + acc_greedy
    print "Dev: Accuracy Viterbi memm : " + acc_viterbi

    print "Evaluation on dev set elapsed: " + str(end - start) + " seconds"
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        start = time.time()
        print "Start evaluation on test set"
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        end = time.time()

        print "Test: Accuracy greedy memm: " + acc_greedy
        print "Test:  Accuracy Viterbi memm: " + acc_viterbi

        print "Evaluation on test set elapsed: " + str(end - start) + " seconds"
        full_flow_end = time.time()
        print "The execution of the full flow elapsed: " + str(full_flow_end - full_flow_start) + " seconds"