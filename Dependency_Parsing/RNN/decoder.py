import numpy as np
import sys
from collections import defaultdict, namedtuple
from operator import itemgetter


def parse_proj(scores, gold=None):
    '''
    Parse using Eisner's algorithm.

    scores - an (n+1) x (n+1) matrix
    gold - the gold arcs
    '''

    #IMPLEMENT YOUR CODE BELOW
    #raise NotImplementedError
    n = scores.shape[0]
    aug_scores = scores

    c = np.zeros((n, n, 2, 2), dtype = np.int)
    bp = np.zeros((n, n, 2, 2), dtype = np.int)
    bp.fill(-1)
    #print(scores)
    if gold is not None:
        for m in range(1, n):
            for i in range(n-m):
                j = i + m
                if gold[j] != i:
                    aug_scores[i][j] = aug_scores[i][j] + 1
                if gold[i] != j:
                    aug_scores[j][i] = aug_scores[j][i] + 1

    for m in range(1, n):
        for i in range(n-m):
        j = i + m

        lst_rincomp = [c[i][k][1][1] + c[k+1][j][0][1] + aug_scores[i][j] for k in range(i, j)]
        lst_rcomp = [c[i][k][1][0] + c[k][j][1][1] for k in range(i+1, j+1)]
        lst_lincomp = [c[i][k][1][1] + c[k+1][j][0][1] + aug_scores[j][i] for k in range(i, j)]
        lst_lcomp = [c[i][k][0][0] + c[k][j][0][1] for k in range(i, j)]

        c[i][j][1][0] = np.amax(lst_rincomp)
        c[i][j][1][1] = np.amax(lst_rcomp)
        c[i][j][0][0] = np.amax(lst_lincomp)
        c[i][j][0][1] = np.amax(lst_lcomp)
        bp[i][j][1][0] = np.argmax(lst_rincomp) + i
        bp[i][j][1][1] = np.argmax(lst_rcomp) + i+1
        bp[i][j][0][0] = np.argmax(lst_lincomp) + i
        bp[i][j][0][1] = np.argmax(lst_lcomp) + i
        #print(("i", i, "j", j, "\n", "bp10",bp[i][j][1][0], "bp11", bp[i][j][1][1], "bp00", bp[i][j][0][0], "bp01", bp[i][j][0][1]))

h = [None] * n
h[0] = -1

def backtrack(bp, i, j, r, c, h):
    if i == j:
        return
    k = bp[i][j][r][c]
    if c == 1:
        if r == 1:
            backtrack(bp, i, k, 1, 0, h)
            backtrack(bp, k, j, 1, 1, h)
        else:
            backtrack(bp, i, k, 0, 1, h)
            backtrack(bp, k, j, 0, 0, h)
    else:
        if r == 1:
            h[j] = i
            else:
                h[i] = j
            backtrack(bp, i, k, 1, 1, h)
            backtrack(bp, k+1, j, 0, 1, h)

    backtrack(bp, 0, n-1, 1, 1, h)
    print(h)
    return h
