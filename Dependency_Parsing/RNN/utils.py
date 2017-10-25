    def normalize(word):
        '''
        takes in a word string and returns the NUMBER token if a number and lowercase otherwise
        '''
        #raise NotImplementedError
        word = word.lower()
        word = re.sub("(-)?[0-9]+(\.[0-9]+)?", NUMBER, word)
        return word

    def process(self, data, deterministic=False):
        '''
        convert a list of ConllEntry to a list of indices for each token type

        data - a list of lists of ConllEntrys
        deterministic - a parameter indicating whether to randomly replace words with the UNKNOWN token

        returns indices, pos indices, parent indices, and dependency relation labels
        '''

        #YOUR IMPLEMENTATION GOES HERE
        #raise NotImplementedError
        indices = []
        pos_indices = []
        par_indices = []
        rel_indices = []
        w_freq = Counter()
        for lst in data:
            for conll in lst:
                w_freq.update([conll.norm])

        for lst in data:
            lst_indices = []
            lst_pos_indices = []
            lst_rel_indices = []
            lst_par_indices = []

            for conll in lst:
                if not deterministic:
                    prob = 0.25/(0.25 + w_freq[conll.norm])
                    replace = np.random.choice(range(0, 2), p=[1-prob, prob])
                    if replace:
                        word = UNKNOWN
                    else:
                        word = conll.norm
                    index = self.word2idx[word]
                try:
                    index = self.word2idx[conll.norm]
                except:
                    index = self.word2idx[UNKNOWN]
                lst_indices.append(index)

                try:
                    pos_index = self.pos2idx[conll.pos]
                except:
                    pos_index = 0
                lst_pos_indices.append(pos_index)

                try:
                    rel_index = self.rel2idx[conll.relation]
                except:
                    rel_index = 0
                lst_rel_indices.append(rel_index)

                par_index = conll.parent_id
                lst_par_indices.append(par_index)

            indices.append(lst_indices)
            pos_indices.append(lst_pos_indices)
            rel_indices.append(lst_rel_indices)
            par_indices.append(lst_par_indices)
        return indices, pos_indices, par_indices, rel_indices
                                                                                                                 164,1-8       55%
