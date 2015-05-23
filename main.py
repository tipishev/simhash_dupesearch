#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, absolute_import

__author__ = 'tipishev'

from os import walk
import md5
from string import punctuation
from re import sub
from random import random, seed as set_random_seed
import logging
from math import sqrt
import matplotlib.pyplot as plt

# filesystem stuff
LOGFILE = "main.log"
DOCS_DIRECTORY = "../docs/"

# algorithm stuff
RANDOM_SEED = 42
NUMBER_OF_DIMENSIONS = 64
NUMBER_OF_VECTORS = 8


def md5sum(string):
    return md5.new(string.encode('utf8')).hexdigest()

def hamming_distance(first, second):
    assert len(first)==len(second)
    return sum([a!=b for (a, b) in zip(first, second)])

def to_words(raw_text):
    return sub('[{}]'.format(punctuation), ' ', raw_text.lower()).split()

def to_text(filename):
    with open(filename, 'r') as f:
        return ' '.join(line.decode('utf8') for line in f)

def to_binary(hexstring, length_in_bits=128):
    return bin(int(hexstring, 16))[2:].zfill(length_in_bits)

def file_to_words(filename):
    return to_words((to_text(filename)))

def to_hash_vector(words, length):
    ''' create a sum vector depending on each word's md5 sum '''
    result = [0 for _ in range(length)]
    for word in words:
        control_vector = to_binary(md5sum(word))[0:length]
        for position, char in enumerate(control_vector):
            result[position] += +1 if char == '1' else -1
    return result


def vector(start, end):
    assert len(start) == len(end)
    return [e-s for e,s in zip(end,start)]

def add(vector1, vector2):
    return [sum(e) for e in zip(vector1, vector2)]

def magnitude(vector):
    return sqrt(sum([e*e for e in vector]))

def twin(vector):
    return [-e for e in vector]

def random_unit_vector(n):
    vector = [random() for _ in range(n)]
    length = magnitude(vector)
    return [v/length for v in vector]

def coulombs_force(pusher, pushee):
    CHARGE1 = CHARGE2 = K = 1
    direction = vector(pusher, pushee)
    force_magnitude = K*CHARGE1*CHARGE2/magnitude(direction)**2
    return [e*force_magnitude for e in direction]

def generate_vectors(m, n, number_of_iterations=5):
    ''' generates m vectors equally spread on an n-dimensional hypersphere '''
    vectors = [random_unit_vector(n) for _ in range(m)]
    for iteration in range(number_of_iterations):
        for vector in vectors:
            total_force = [0 for _ in range(len(vector))]
            for other in [v for v in vectors if v != vector]:
                force = coulombs_force(other, vector)
                total_force = add(total_force, force)
            vector = add(vector, total_force)
    return vectors

def plot_vectors(vectors):
    x, y = tuple(zip(*vectors))
    plt.plot(x, y, 'ro')
    plt.axis([-2, 2, -2, 2])
    plt.show()

def main():
    vectors = generate_vectors(3, 2)
    plot_vectors(vectors)
    # pusher = [0.3, 0.2]
    # pushee = [0.6, 0.1]
    # force = coulombs_force(pusher, pushee)
    # print(force)
    # plot_vectors([pusher, pushee, force, [0,0]])

    # print(to_hash_vector(file_to_words('article'), NUMBER_OF_VECTORS))

if __name__ == '__main__':
    logging.basicConfig(filename=LOGFILE, level=logging.INFO)
    log = logging.getLogger(__name__)
    set_random_seed(RANDOM_SEED)
    main()
