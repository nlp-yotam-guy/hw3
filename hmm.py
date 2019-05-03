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
            add_to_dict((sentence[j][0], sentence[j][1]), e_word_tag_counts)
            add_to_dict(sentence[j][1], e_tag_counts)
            total_tokens += 1
        if len(sentence) > 1:
            add_to_dict((sentence[0][1], sentence[1][1]), q_bi_counts)
            add_to_dict((sentence[1][1]), q_uni_counts)
            add_to_dict((sentence[1][0], sentence[1][1]), e_word_tag_counts)
            add_to_dict(sentence[1][1], e_tag_counts)
            total_tokens += 1
        add_to_dict((sentence[0][1]), q_uni_counts)
        add_to_dict((sentence[0][0], sentence[0][1]), e_word_tag_counts)
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

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    sent.append(('STOP', 'STOP'))
    pi = dict()
    bp = dict()
    S = dict()
    n = len(sent)
    #Initialization to S
    S[-2] = {'*'}
    S[-1] = {'*'}
    for k in range(n):
        #S[k] = get_all_tags(e_word_tag_counts, sent[k])
        S[k] = e_tag_counts.keys()
    #base case
    pi[(-1, '*', '*')] = 1.0

    for k in range(n):
        for u in S[k-1]:
            for v in S[k]:
                prob_max = 0
                bp_max = ''
                for w in S[k-2]:
                    if (k-1,w,u) not in pi:
                        prob = 0
                    else:
                        q = Calculate_q_ML(w,u,v,lambda1,lambda2,q_tri_counts,q_bi_counts,q_uni_counts,total_tokens)
                        prob = pi[(k-1,w,u)] * q
                    if prob > prob_max:
                        prob_max = prob
                        bp_max = w
                pi[(k,u,v)] = prob_max
                bp[(k - 1, w, u)] = bp_max
    prob_max = 0
    u_max = ''
    v_max = ''
    for u in S[n - 3]:
        for v in S[n-2]:
            q = Calculate_q_ML(u,v,'STOP',lambda1,lambda2,q_tri_counts,q_bi_counts,q_uni_counts,total_tokens)
            if (n-2,u,v) not in pi:
                prob = 0
            else:
                prob = pi[(n-2,u,v)]*q
            if prob > prob_max:
                prob_max = prob
                u_max = u
                v_max = v
    predicted_tags[n-2] = v_max
    predicted_tags[n-3] = u_max

    for k in range(n-4,-1,-1):
        predicted_tags[k] = bp[(k+2,predicted_tags[k+1],predicted_tags[k+2])]
    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print "Start evaluation"
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    errors = 0
    lambda1 = 0.5
    lambda2 = 0.4
    counter = 0
    for sentence in test_data:
        tags_from_viterbi = hmm_viterbi(sentence,total_tokens,q_tri_counts,q_bi_counts,q_uni_counts,e_word_tag_counts,e_tag_counts,lambda1,lambda2)
        for i in range(len(tags_from_viterbi)):
            if sentence[i][1] != tags_from_viterbi[i]:
                errors+=1
            counter+=1
    acc_viterbi = 1 - errors/counter
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