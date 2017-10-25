import sys
import nltk
import math
import time

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(Brown_train):
    brown_words = []
    brown_tags = []

    for sentence in Brown_train:
	words_list = []
	tags_list = []
	for i in range(2):
	    words_list.append(START_SYMBOL)
	    tags_list.append(START_SYMBOL)
	for wordtag in (sentence[:-2] + STOP_SYMBOL + '/' + STOP_SYMBOL).split(' '):
	    def positionlast(x, s):
		return next(i for i,j in list(enumerate(s))[::-1] if j==x)
	    position = positionlast('/', wordtag)
	    words_list.append(wordtag[0:position])
	    tags_list.append(wordtag[(position+1):])
	brown_words.append(words_list)
	brown_tags.append(tags_list)

    return brown_words, brown_tags

# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    count_unigram = {}
    count_bigram = {}
    count_trigram = {}
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    N = 0

    for tags in brown_tags:
	unigram_tags = tags[2:]
	bigram_tags = list(nltk.bigrams(tags[1:]))
	trigram_tags = list(nltk.trigrams(tags))
	for item in set(unigram_tags):
	    count_unigram[item] = float(count_unigram.get(item, 0) + unigram_tags.count(item))
	for item in set(bigram_tags):
	    count_bigram[item] = float(count_bigram.get(item, 0) + bigram_tags.count(item))
	for item in set(trigram_tags):
	    count_trigram[item] = float(count_trigram.get(item, 0) + trigram_tags.count(item))
	N += len(unigram_tags)

    for key in count_unigram:
	unigram_p[key] = math.log(count_unigram[key]/N, 2)
    for key in count_bigram:
	if key[0] == START_SYMBOL:
	    bigram_p[key] = math.log(count_bigram[key]/(len(brown_tags)), 2)
	else:
	    bigram_p[key] = math.log(count_bigram[key]/count_unigram[key[0]], 2)
    for key in count_trigram:
	if key[0:2] == (START_SYMBOL, START_SYMBOL):
	    trigram_p[key] = math.log(count_trigram[key]/len(brown_tags), 2)
	else:
	    trigram_p[key] = math.log(count_trigram[key]/count_bigram[key[0:2]], 2)

    q_values = trigram_p
    print "NOUN DET NOUN", q_values[("NOUN", "DET", "NOUN")]
    print "* * ADJ", q_values[('*', '*', 'ADJ')]
    print "X . STOP", q_values[("X", ".", "STOP")]
    print "TRIGRAM CONJ ADV NOUN", q_values[("CONJ", "ADV", "NOUN")]
    print "TRIGRAM DET NUM NOUN", q_values[("DET", "NUM", "NOUN")]
    print "TRIGRAM NOUN PRT CONJ", q_values[("NOUN", "PRT", "CONJ")]
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    count_words = {}
    for sentence in brown_words:
 	for word in set(sentence):
	    count_words[word] = count_words.get(word, 0) + sentence.count(word)
    known_words = [key for (key, value) in count_words.items() if value > RARE_WORD_MAX_FREQ]
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    d = {}

    for word in known_words:
	d[word] = 1

    for sentence in brown_words:
        word_list = []
	    for word in sentence:
            try:
    		    value = d[word]
    		    word_list.append(word)
    	    except:
    		    word_list.append(RARE_SYMBOL)
	brown_words_rare.append(word_list)

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = []
    count_tag = {}
    count_tuple = {}
    brown_tuples = [zip(brown_words_rare[i], brown_tags[i]) for i in xrange(len(brown_tags))]

    for tags in brown_tags:
	    for tag in set(tags):
	        count_tag[tag] = float(count_tag.get(tag, 0) + tags.count(tag))
            taglist.append(tag)
        taglist = set(taglist)

    for tuples in brown_tuples:
	    for t in set(tuples):
	        count_tuple[t] = float(count_tuple.get(t, 0) + tuples.count(t))

    for key in count_tuple:
	    e_values[key] = math.log(count_tuple[key], 2) - math.log(count_tag[key[1]], 2)

    print "York NOUN", e_values[("York", "NOUN")]
    print "* *", e_values[("*", "*")]
    print "midnight NOUN", e_values[("midnight", "NOUN")]
    print "Place VERB", e_values[("Place", "VERB")]
    print "primary ADJ", e_values[("primary", "ADJ")]
    print "STOP STOP", e_values[("STOP", "STOP")]
    print "_RARE_ VERB", e_values[("_RARE_", "VERB")]
    print "_RARE_ X", e_values[("_RARE_", "X")]
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

def forward(brown_dev_words,taglist, known_words, q_values, e_values):
    probs = []
    taglist.remove(START_SYMBOL)
    taglist.remove(STOP_SYMBOL)
    brown_dev_rare_words = replace_rare(brown_dev_words, known_words)

    def logsum(x, y):
	log_s = math.log((1 + 2**(y-x)), 2) + x
	return log_s

    for sentence in brown_dev_rare_words:
        pi = {}
        s = [START_SYMBOL]
        length = len(sentence)

	for tag in taglist:
	    try:
		q = q_values[(START_SYMBOL, START_SYMBOL, tag)]
	    except:
		q = LOG_PROB_OF_ZERO
	    try:
	        pi[(1, START_SYMBOL, tag)] = q + e_values[(sentence[0], tag)]
	    except:
		continue

        for i in range(length):
	    s.append(taglist)

        for i in range(2, length+1):
            for tag3 in s[i]:
	        for tag2 in s[i-1]:
	  	    for tag1 in s[i-2]:
			try:
			    q = q_values[(tag1, tag2, tag3)]
			except:
			     q = LOG_PROB_OF_ZERO
		      	try:
			    pi[(i, tag2, tag3)] = logsum(pi[(i-1, tag1, tag2)]+q+e_values[(sentence[i-1], tag3)], pi.get((i, tag2, tag3),LOG_PROB_OF_ZERO))
			except:
		            continue

        for tag_2 in s[length]:
	        for tag_1 in s[length-1]:
		        try:
		            q = q_values[(tag_1, tag_2, STOP_SYMBOL)]
		        except:
		            q = LOG_PROB_OF_ZERO
	            try:
                    pi[(length+1, tag_2, STOP_SYMBOL)] = logsum(pi.get((length+1, tag_2, STOP_SYMBOL), LOG_PROB_OF_ZERO), pi[(length, tag_1, tag_2)] + q)
		        except:
	                continue

	for tag in taglist:
	    try:
	        pi[(length+1, STOP_SYMBOL)] = logsum(pi.get((length+1, STOP_SYMBOL), LOG_PROB_OF_ZERO), pi[(length+1, tag, STOP_SYMBOL)])
	    except:
		continue
	try:
            probs.append(str(pi[(length+1, STOP_SYMBOL)])+'\n')
	except:
	    probs.append(str(LOG_PROB_OF_ZERO)+'\n')
	#if brown_dev_rare_words.index(sentence) > 4:
	#    break

    print(probs[0:5])
    return probs

# This function takes the output of forward() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    brown_dev_rare_words = replace_rare(brown_dev_words, known_words)

    for m in range(len(brown_dev_words)):
	pi = {}
	bp = {}
	length = len(brown_dev_words[m])

	for tag in taglist:
	    pi[(0, START_SYMBOL, tag)] = q_values.get((START_SYMBOL, START_SYMBOL, tag), LOG_PROB_OF_ZERO) + e_values.get((brown_dev_rare_words[m][0], tag), LOG_PROB_OF_ZERO)
	    bp[(0, START_SYMBOL, tag)] = START_SYMBOL

        for tag2 in taglist:
      	    for tag1 in taglist:
		pi[(1, tag1, tag2)] = pi.get((0, START_SYMBOL, tag1), LOG_PROB_OF_ZERO) + q_values.get((START_SYMBOL, tag1, tag2), LOG_PROB_OF_ZERO) + e_values.get((brown_dev_rare_words[m][1], tag2), LOG_PROB_OF_ZERO)
		bp[(1, tag1, tag2)] = START_SYMBOL

	for i in range(2, length):
	    for tag3 in taglist:
	        for tag2 in taglist:
		    log_pb = float('-Inf')
	    	    for tag1 in taglist:
		        pi[(i, tag2, tag3)] = pi.get((i-1, tag1, tag2), LOG_PROB_OF_ZERO) + q_values.get((tag1, tag2, tag3), LOG_PROB_OF_ZERO) + e_values.get((brown_dev_rare_words[m][i], tag3), LOG_PROB_OF_ZERO)
		    	if pi[(i, tag2, tag3)] > log_pb :
			    log_pb = pi[(i, tag2, tag3)]
			    tag_max = tag1
		    pi[(i, tag2, tag3)] = log_pb
		    bp[(i, tag2, tag3)] = tag_max


	log_pb = float('-Inf')
	tag_max = ''
	for tag_2 in taglist:
	    for tag_1 in taglist:
   	        pi[(length, tag_2, STOP_SYMBOL)] = pi.get((length-1, tag_1, tag_2), LOG_PROB_OF_ZERO) + q_values.get((tag_1, tag_2, STOP_SYMBOL), LOG_PROB_OF_ZERO)
		if pi[(length, tag_2, STOP_SYMBOL)] > log_pb:
		    log_pb = pi[(length, tag_2, STOP_SYMBOL)]
		    tag_max = (tag_1, tag_2)

	tags = []
	tags.append(tag_max[1])
	tags.append(tag_max[0])
	for j in range(1, length-1):
	    tags.append(bp[(length-j, tags[j], tags[j-1])])

	sentence_tagged = ''
	for k in range(length):
	    sentence_tagged = sentence_tagged + brown_dev_words[m][k] + '/' + tags[length-1-k] +' '
	print(sentence_tagged)
	tagged.append(sentence_tagged+'\n')
	#if m > 1:
	#    break
	#print(tagged[0:2])
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens.
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff = default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff = bigram_tagger)

    for sentence in brown_dev_words:
	string = ""
        sent = trigram_tagger.tag(sentence)
	for tuple in sent:
	    string = string + tuple[0] + '/' + tuple[1] + ' '
	tagged.append(string + '\n')

    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q7_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare



    # open Brown development data (question 6)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # question 5
    forward_probs = forward(brown_dev_words,taglist, known_words, q_values, e_values)
    q5_output(forward_probs, OUTPUT_PATH + 'B5.txt')

    # do viterbi on brown_dev_words (question 6)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 6 output
    q6_output(viterbi_tagged, OUTPUT_PATH + 'B6.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 7 output
    q7_output(nltk_tagged, OUTPUT_PATH + 'B7.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
