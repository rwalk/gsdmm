from unittest import TestCase
from gsdmm.mgp import MovieGroupProcess
import numpy

class TestGSDMM(TestCase):
    '''This class tests the Panel data structures needed to support the RSK model'''

    def setUp(self):
        numpy.random.seed(47)

    def tearDown(self):
        numpy.random.seed(None)

    def compute_V(self, texts):
        V = set()
        for text in texts:
            for word in text:
                V.add(word)
        return len(V)

    def test_grades(self):

        grades = list(map(list, [
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "D",
            "D",
            "F",
            "F",
            "P",
            "W"
        ]))

        grades = grades + grades + grades + grades + grades
        mgp = MovieGroupProcess(K=100, n_iters=100, alpha=0.001, beta=0.01)
        y = mgp.fit(grades, self.compute_V(grades))
        self.assertEqual(len(set(y)), 7)
        for words in mgp.cluster_word_distribution:
            self.assertTrue(len(words) in {0,1}, "More than one grade ended up in a cluster!")
            
    def test_simple_example(self):
        # example from @spattanayak1

        docs=[['house',
        'burning',
        'need',
        'fire',
        'truck',
        'ml',
        'hindu',
        'response',
        'christian',
        'conversion',
        'alm']]

        mgp = MovieGroupProcess(K=10, alpha=0.1, beta=0.1, n_iters=30)

        vocab = set(x for doc in docs for x in doc)
        n_terms = len(vocab)
        n_docs = len(docs)

        y = mgp.fit(docs, n_terms)

    def test_short_text(self):
        # there is no perfect segmentation of this text data:
        texts = [
            "where the red dog lives",
            "red dog lives in the house",
            "blue cat eats mice",
            "monkeys hate cat but love trees",
            "green cat eats mice",
            "orange elephant never forgets",
            "orange elephant must forget",
            "monkeys eat banana",
            "monkeys live in trees",
            "elephant",
            "cat",
            "dog",
            "monkeys"
        ]

        texts = [text.split() for text in texts]
        V = self.compute_V(texts)
        mgp = MovieGroupProcess(K=30, n_iters=100, alpha=0.2, beta=0.01)
        y = mgp.fit(texts, V)
        self.assertTrue(len(set(y))<10)
        self.assertTrue(len(set(y))>3)
