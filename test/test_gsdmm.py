from unittest import TestCase
from gsdmm.mgp import MovieGroupProcess
import numpy


def tokenize(docs):
    words = set()
    for doc in docs:
        for word in doc:
            words.add(word)
    tokens = {word: i for i, word in enumerate(words)}
    return [[tokens[word] for word in doc] for doc in docs]


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
        grades = [g for _ in range(5) for g in grades]
        mgp = MovieGroupProcess(K=100, n_iters=100, alpha=1e2, beta=1e-6)
        y = mgp.fit(tokenize(grades), self.compute_V(grades))
        self.assertEqual(len(set(y)), 7)
        distribution = mgp.cluster_word_distribution
        for label in range(distribution.shape[0]):
            tmp = distribution[label, :]
            self.assertTrue(numpy.all(tmp == 0) or numpy.sum(tmp != 0) == 1,
                            f"More than one grade ended up in a cluster!")

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
        y = mgp.fit(tokenize(texts), V)
        self.assertTrue(len(set(y))<10)
        self.assertTrue(len(set(y))>3)