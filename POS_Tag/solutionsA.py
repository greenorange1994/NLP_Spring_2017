import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    count_unigram = {}
    count_bigram = {}
    count_trigram = {}
    N = 0

    for sentence in training_corpus:
	tokens_unigram = []
	tokens_bigram = [START_SYMBOL]
	tokens_trigram = [START_SYMBOL, START_SYMBOL]
	for token in (sentence[:-2] + STOP_SYMBOL).split(' '):
	    tokens_unigram.append((token, ' '))
	    tokens_bigram.append(token)
	    tokens_trigram.append(token)
	unigram_tuples = tokens_unigram
        bigram_tuples = list(nltk.bigrams(tokens_bigram))
	trigram_tuples = list(nltk.trigrams(tokens_trigram))
        for item in set(unigram_tuples):
	    count_unigram[item] = float(count_unigram.get(item, 0) + unigram_tuples.count(item))
	for item in set(bigram_tuples):
	    count_bigram[item] = float(count_bigram.get(item, 0) + bigram_tuples.count(item))
	for item in set(trigram_tuples):
	    count_trigram[item] = float(count_trigram.get(item, 0) + trigram_tuples.count(item))
	N += len(tokens_unigram)

    for key in count_unigram:
	unigram_p[key] = math.log(count_unigram[key]/N, 2)
    for key in count_bigram:
	if key[0] == START_SYMBOL:
	    bigram_p[key] = math.log(count_bigram[key]/len(training_corpus), 2)
	else:
	    bigram_p[key] = math.log(count_bigram[key]/count_unigram[(key[0], ' ')], 2)
    for key in count_trigram:
	if key[0:2] == (START_SYMBOL, START_SYMBOL):
	    trigram_p[key] = math.log(count_trigram[key]/len(training_corpus), 2)
	else:
            trigram_p[key] = math.log(count_trigram[key]/count_bigram[key[0:2]], 2)

    print("near", unigram_p[("near", " ")])
    print("near the", bigram_p[("*", "Captain")])
    print("near the ecliptic", trigram_p[("near", "the", "ecliptic")])
    print('and not come', trigram_p[("really", "up", "for")])
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
#tr(unigrams[unigram]) + '\n')
#Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc.
def score(ngram_p, n, corpus):

    index = 0
    scores = []

    for sentence in corpus:
	tokens_ngram = []
	scores.append(0)
	for i in range(n):
	    if i == 0 :
		continue
	    else:
	        tokens_ngram.append(START_SYMBOL)
	for token in (sentence[:-2] + STOP_SYMBOL).split(' '):
	    if n == 1 :
		tokens_ngram.append((token, ' '))
	    else :
		tokens_ngram.append(token)
	if n == 1 :
	    ngram_tuples = tokens_ngram
	elif n == 2 :
	    ngram_tuples = list(nltk.bigrams(tokens_ngram))
	else :
	    ngram_tuples = list(nltk.trigrams(tokens_ngram))
	for t in ngram_tuples:
	    try:
	        scores[index] = scores[index] + ngram_p[t]
	    except:
		    scores[index] = MINUS_INFINITY_SENTENCE_LOG_PROB
		    break  
	index += 1

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

#TODO: IMPLEMENT THIS FUNCTION
# Calculcates the perplexity of a language model
# scores_file: one of the A2 output files of scores
# sentences_file: the file of sentences that were scores (in this case: data/Brown_train.txt)
# This function returns a float, perplexity, the total perplexity of the corpus
def calc_perplexity(scores_file, sentences_file):

    f_scores = open(scores_file, 'r')
    f_sentences = open(sentences_file, 'r')
    scores = f_scores.readlines()
    corpus = f_sentences.readlines()
    f_scores.close()
    f_sentences.close()

    pp = 0
    N = 0
    for sentence in corpus:
	N += len(sentence.split(' '))

    for score in scores:
	score = float(score)
	pp += score

    perplexity = 2**(-pp/N)

    return perplexity

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    linear_p = {}
    lambdas = 1.0/3
    index = 0
    for tokens in corpus:
	tokens_trigram = [START_SYMBOL, START_SYMBOL]
	scores.append(0)
	for token in (tokens[:-2] + STOP_SYMBOL).split():
	    tokens_trigram.append(token)
	trigram_tuples = list(nltk.trigrams(tokens_trigram))
	for t in trigram_tuples:
	    try:
		uni_p = 2**unigrams[(t[2], ' ')]
	    except:
		uni_p = 0
	    try:
		bi_p = 2**bigrams[(t[1], t[2])]
	    except:
		bi_p = 0
	    try:
		tri_p = 2**trigrams[t]
	    except:
		tri_p = 0
	    try:
	        linear_p[t] = math.log(lambdas * (tri_p + bi_p + uni_p), 2)
	        scores[index] = scores[index] + linear_p[t]
	    except:
		scores[index] = MINUS_INFINITY_SENTENCE_LOG_PROB
		break
	index += 1
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
