from providedcode.transitionparser import TransitionParser
from providedcode.dependencygraph import DependencyGraph
import sys

if __name__ == '__main__':

    tp = TransitionParser.load(sys.argv[1])
    sentences = sys.stdin.readlines()

    for sentence in sentences:
	data = DependencyGraph.from_sentence(sentence)
	p = tp.parse([data])
        print p[0].to_conll(10).encode('utf-8')

