from data import *

def incrementpos(word_to_pos_count, token):
    if token[1] in word_to_pos_count[token[0]]:
        word_to_pos_count[token[0]][token[1]] += 1
    else:
        word_to_pos_count[token[0]][token[1]] = 1


def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    ### YOUR CODE HERE
    word_to_pos_count = dict()
    word_to_max_pos = dict()
    for sent in train_data:
       for token in sent:
            if token[0] in word_to_pos_count:
                incrementpos(word_to_pos_count, token)
            else:
                word_to_pos_count[token[0]] = dict()
                incrementpos(word_to_pos_count, token)
    for word in word_to_pos_count:
        #pos_max = max(word_to_pos_count[word])
        pos_max= max(word_to_pos_count[word], key=word_to_pos_count[word].get)
        word_to_max_pos[word] = pos_max
    return word_to_max_pos
    ### END YOUR CODE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    ### YOUR CODE HERE
    errors = 0
    tokens=0
    for test_sent in test_set:
        for token in test_sent:
            if pred_tags[token[0]] != token[1]:
                errors += 1
            tokens += 1
    return 1 - float(errors)/tokens

    ### END YOUR CODE

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + str(most_frequent_eval(dev_sents, model))

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + str(most_frequent_eval(test_sents, model))