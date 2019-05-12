from data import *
import time

def add_to_dict(key,dict):
    if key in dict:
        dict[key] += 1
        return 0
    else:
        dict[key] = 1
        return 1

def Calculate_q_ML(word1,word2,word3,lambda1,lambda2,trigram_counts,bigram_counts,unigram_counts,train_token_count):
    # Calculate q_ML of the trigram
    if (word1, word2, word3) in trigram_counts:
        q_tri = float(trigram_counts[(word1, word2, word3)]) / bigram_counts[
        (word1, word2)]
    else:
        q_tri = 0
    # Calculate q_ML of the bigram
    if (word2, word3) in bigram_counts:
        q_bi = float(bigram_counts[(word2, word3)]) / unigram_counts[word2]
    else:
        q_bi = 0
    # Calculate q_ML of the unigram

    if word3 in unigram_counts:
        q_uni = float(unigram_counts[word3]) / train_token_count
    else:
        q_uni = 0

    # Calculate  the linear interpolation
    q = (lambda1 * q_tri + lambda2 * q_bi + float((1.0 - lambda1 - lambda2)) * q_uni)
    return q

def add_to_e_word_tag_counts(word,tag,e_word_tag_dict):
    if word not in e_word_tag_dict:
        e_word_tag_dict[word] = dict()
    add_to_dict(tag, e_word_tag_dict[word])


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print "Start training"
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts = {}, {}, {}, {}, {}
    ### YOUR CODE HERE
    for sentence in sents:
        sentence.append(('STOP', 'STOP'))
        for j in range(2, len(sentence)):
            add_to_dict((sentence[j - 2][1], sentence[j - 1][1], sentence[j][1]), q_tri_counts)
            add_to_dict((sentence[j - 1][1], sentence[j][1]), q_bi_counts)
            add_to_dict((sentence[j][1]), q_uni_counts)
            add_to_e_word_tag_counts(sentence[j][0],sentence[j][1],e_word_tag_counts)
            #add_to_dict((sentence[j][0], sentence[j][1]), e_word_tag_counts)
            add_to_dict(sentence[j][1], e_tag_counts)
            total_tokens += 1
        if len(sentence) > 1:
            add_to_dict((sentence[0][1], sentence[1][1]), q_bi_counts)
            add_to_dict((sentence[1][1]), q_uni_counts)
            add_to_dict((sentence[1][0], sentence[1][1]), e_word_tag_counts)
            add_to_dict(sentence[1][1], e_tag_counts)
            total_tokens += 1
        add_to_dict((sentence[0][1]), q_uni_counts)
        add_to_e_word_tag_counts(sentence[0][0], sentence[1][1], e_word_tag_counts)
        #add_to_dict((sentence[0][0], sentence[0][1]), e_word_tag_counts)
        add_to_dict(sentence[0][1], e_tag_counts)
        total_tokens += 1
    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts

def get_all_tags(e_word_tag_counts, word):
    s = set()
    for key in e_word_tag_counts:
        if key[0] == word[0]:
            s.add(key[1])
    return s

def prune(word, tags):
    word = word.lower()
    if word == ',':
        return {','}
    if word == '.':
        return {'.'}
    if word == 'and':
        return {'CC', 'NN'}
    if word == ':':
        return {':'}
    if word == '#':
        return {'#'}
    if word == '$':
        return {'$'}
    if word.endswith('tion'):
        return {'NN', 'NNP'}
    if word == '_nounlike_':
        return {'NNP', 'NNS','JJ','NN'}
    if word == 'to':
        return {'TO', 'NN'}
    if word == 'from':
        return {'CC', 'NN', 'IN'}
    if word == 'with':
        return {'NN', 'IN'}
    if word in {'on', 'by', 'at'}:
        return {'RP', 'IN'}
    if word == 'the':
        return {'DT'}
    if word == '``':
        return {'``'}
    if word == 'a':
        return {'IN', 'DT'}
    if word == 'in':
        return {'IN'}
    if word == 'about':
        return {'IN', 'RB'}
    if word in {'what', 'how', 'when', 'who', 'whom', 'where', 'which', 'whose', 'why'} or word == 'w_h':
        return {'WDT', 'WP', 'WP$', 'WRB'}
    if word == 'but' or word == 'coordinating_conjunction':
        return {'CC'}
    if word == 'that':
        return {'CC', 'IN', 'DT', 'WDT'}
    if word in {'one', 'two', 'three', 'four', 'five'}:
        return {'CD'}
    if word == 'up':
        return {'IN', 'RB', 'RP'}
    if word in {'over', 'under'}:
        return {'IN', 'RB', 'JJ'}
    if word == '-lrb-':
        return {'-LRB-'}
    if word in {'i', 'he', 'she', 'you', 'it'} or word == 'personal_pronoun':
        return {'PRP'}
    if word == 'allcaps':
        return {'NN', 'JJ', 'NNS', 'NNP', 'VBD', 'RB'}
    if word == '_verblike_':
        return {'NNS','JJ','VBZ','VB','VBD','VBN'}
    if word == 'ing':
        return {'VBP','VBG','NNS','JJ'}
    if word == 'superlative_like':
        return {'JJS'}
    if word == 'comparative_like' or word.endswith('ier'):
        return {'JJR','NNP','NN','JJ','RBR'}
    if word == 'determines':
        return {'DT'}
    if word == 'adverb':
        return {'RB'}
    if word == 'day':
        return {'NN','RB'}
    if word == '_adjlike_':
        return {'JJ','NNS','NNP','VBN','VBD','VB','VBZ'}
    if word == 'as':
        return {'RB', 'IN','CC'}
    if word == '_verblike_':
        return {'VB','VBN'}
    if word == 'company':
        return {'JJ','NN','NNP','NNS','VBG'}
    if word =='person':
        return {'NN'}
    if word in {'yet', 'also', 'because','unless'}:
        return {'RB','CC','IN'}

    return tags

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    import random
    ### YOUR CODE HERE
    sent.append(('STOP', 'STOP'))
    pi = dict()
    bp = dict()
    S = dict()
    n = len(sent)
    #Initialization to S
    S[-2] = {'*'}
    S[-1] = {'*'}
    #for k in range(n-1):
    #    S[k] = get_all_tags(e_word_tag_counts, sent[k])
    #    S[k] = prune(S[k],sent[k][0])
    #     #S[k] = e_tag_counts.keys()
    S[n-1] = {'STOP'}
    #base case
    pi[(-1, '*', '*')] = 1.0
    for k in range(n):
        if k <= n-2:
            word = sent[k][0]
            if sent[k][0] in e_word_tag_counts:
                S[k] = set(e_word_tag_counts[sent[k][0]])
                S[k] = prune(sent[k][0], S[k])
            else:
                S[k] = set(e_tag_counts)
        end_max_prob= float('-Inf')
        for u in S[k-1]:
            for v in S[k]:
                prob_max = float('-Inf')
                bp_max = None
                for w in S[k-2]:
                    if (k-1,w,u) not in pi:
                        prob = 0
                    else:
                        if sent[k][0] not in e_word_tag_counts or v not in e_word_tag_counts[sent[k][0]]:
                            e=0
                        else:
                            e = float(e_word_tag_counts[sent[k][0]][v])/e_tag_counts[(sent[k][1])]
                        q = Calculate_q_ML(w, u, v,lambda1,lambda2,q_tri_counts,q_bi_counts,q_uni_counts,total_tokens)
                        prob = pi[(k-1,w,u)] * q *e
                    if prob > prob_max:
                        prob_max = prob
                        bp_max = w
                pi[(k, u, v)] = prob_max
                bp[(k, u, v)] = bp_max
                if k == n-1 and pi[(k-1,w,u)] > end_max_prob:
                    end_max_prob = pi[(k-1,w,u)]

    prob_max = float('-Inf')
    u_max = None
    v_max = None
    for u in S[n - 3]:
        for v in S[n-2]:
           q = Calculate_q_ML(u,v,'STOP',lambda1,lambda2,q_tri_counts,q_bi_counts,q_uni_counts,total_tokens)
           prob = pi[(n-2,u,v)] *q
           if prob > prob_max:
               prob_max = prob
               u_max = u
               v_max = v

    predicted_tags[n-2] = v_max
    predicted_tags[n-3] = u_max
    max_tag_count = max(e_tag_counts, key=e_tag_counts.get)
    for i in range(n-4,-1,-1):
            predicted_tags[i] = bp[(i+2,predicted_tags[i+1],predicted_tags[i+2])]
        #else:
        #    test1 = []
        #    for key in bp.keys():
        #        if key.__contains__(i+2):
        #            test1.append(key)
        #    predicted_tags[i] = max_tag_count
    ### END YOUR CODE
    sent.remove(('STOP','STOP'))
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print "Start evaluation"
    acc_viterbi = 0.0
    ### YOUR CODE HERE

    lambdas1 = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}
    lambdas2 = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}
    #lambdas1 = {0.8}
    #lambdas2 = {0.1}
    print_every = 150
    n = len(test_data)
    for lambda1 in lambdas1:
        for lambda2 in lambdas2:
            if lambda1 + lambda2 < 1:
                errors = 0
                counter = 0
                errors_for_test = {}
                for j in range(n):
                    tags_from_viterbi = hmm_viterbi(test_data[j], total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                                    e_word_tag_counts, e_tag_counts, lambda1, lambda2)
                    for i in range(len(tags_from_viterbi)):
                        if test_data[j][i][1] != tags_from_viterbi[i]:
                            #add_to_dict(test_data[j][i][0],  errors_for_test)
                            #print test_data[j][i][0],test_data[j][i][1], tags_from_viterbi[i]
                            errors += 1
                        counter += 1

                #sorted_errors_for_test = sorted(errors_for_test.items(), key=lambda kv: kv[1])
                #print sorted_errors_for_test[-10:]
                acc = 1 - float(errors) / counter
                if acc > acc_viterbi:
                    acc_viterbi = acc
                print "lambda1 = " + str(lambda1) + ", lambda2 = " + str(lambda2) + " accuracy: " + str(acc)


    ### END YOUR CODE

    return acc_viterbi

if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    print "HMM DEV accuracy: " + str(acc_viterbi)

    train_dev_end_time = time.time()
    print "Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds"

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "HMM TEST accuracy: " + str(acc_viterbi)
        full_flow_end_time = time.time()
        print "Full flow elapsed: " + str(full_flow_end_time - start_time) + " seconds"