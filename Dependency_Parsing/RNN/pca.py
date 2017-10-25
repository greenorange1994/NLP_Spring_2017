from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import numpy as np
import utils
import re

if __name__ == "__main__":
    embed = json.loads(open('output/model_pos_embeddings.json').read())
    words = embed["word"]
    postags = embed["pos"]

    #filter out verbs
    train = list(utils.read_conll('data/english/train.conll'))
    word_dict = dict()
    for lst in train:
	for conll in lst:
	    try:
	        word_pos = word_dict[conll.norm]
		word_pos.append(conll.pos)
		word_dict[conll.norm] = word_pos
	    except:
		word_dict[conll.norm] = [conll.pos]
    verb = dict()
    for key, value in words.items():
	regex = r"VB[A-Z]"
	match = []
	try:
	    for postag in word_dict[key]:
	        try:
	            match.append(re.search(regex, postag).group(0))
	        except:
		    continue
	except: #in case of unknown words
	    continue
	if len(match) > 0:
	    verb[key] = value
    
    #turn the list into matrix   
    pos_lst = [value for value in postags.values()]
    pos_mat = np.array(pos_lst)	
    verb_lst = [value for value in verb.values()]
    verb_mat = np.array(verb_lst)
    
    #dimension reduction
    pca = PCA(n_components=2)
    pca.fit(pos_mat)
    pos_mat_pca = pca.fit_transform(pos_mat)
    pca.fit(verb_mat)
    verb_mat_pca = pca.fit_transform(verb_mat)
    
    #plot - pos
    x_pos = [array[0] for array in pos_mat_pca]
    x_pos = np.array(x_pos)
    y_pos = [array[1] for array in pos_mat_pca]
    y_pos = np.array(y_pos)
    plt.figure(figsize=(8,6), dpi=80)
    plt.plot(x_pos, y_pos, "o")
    poskeys = [key for key in postags.keys()]
    for i, tag in enumerate(poskeys):
	plt.annotate(tag, (x_pos[i], y_pos[i]))
    plt.savefig("pos_visualization.png", dpi = 100)

    #plot - verbs
    x_verb = [array[0] for array in verb_mat_pca]
    x_verb = np.array(x_verb)
    y_verb = [array[1] for array in verb_mat_pca]
    y_verb = np.array(y_verb)
    plt.figure(figsize=(6,4), dpi=80)
    plt.plot(x_verb, y_verb, "o")
    verbkeys = [key for key in verb.keys()]
    for i, verb in enumerate(verbkeys):
	plt.annotate(verb, (x_verb[i], y_verb[i]))
    plt.savefig("verb_visualization.png", dpi = 72)

