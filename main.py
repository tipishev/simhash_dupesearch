#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, absolute_import

__author__ = 'tipishev'

# from os import walk
import md5
from string import punctuation
from re import sub
from random import random, seed as set_random_seed
import logging
from math import sqrt
import matplotlib.pyplot as plt
from copy import deepcopy

# filesystem stuff
LOGFILE = "main.log"
DOCS_DIRECTORY = "../docs/"

# algorithm stuff
RANDOM_SEED = 42
NUMBER_OF_DIMENSIONS = 64
NUMBER_OF_VECTORS = 8


def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [(a, b) for a in A for b in B]


def list_without(list_, index):
    return list_[:index] + list_[(index+1):]


def md5sum(string):
    return md5.new(string.encode('utf8')).hexdigest()


def hamming_distance(first, second):
    assert len(first) == len(second)
    return sum([a != b for (a, b) in zip(first, second)])


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

# vector stuff


def create_zero_vector(size):
    return [0 for _ in range(size)]


def normalize(vector):
    return(scale(vector, 1/get_length(vector)))


def to_vector(point1, point2):
    assert len(point1) == len(point2)
    return [e2-e1 for e1, e2 in zip(point1, point2)]


def add(vector1, vector2):
    return [sum(e) for e in zip(vector1, vector2)]


def scale(vector, k):
    return [k*e for e in vector]


def get_length(vector):
    return sqrt(sum([e*e for e in vector]))


def reflect(vector):
    return [-e for e in vector]

# randomization magic


def get_random_vector(n):
    vector = [random() for _ in range(n)]
    return vector

def get_coulombs_force(pusher, pushee):
    CHARGE1 = CHARGE2 = 1
    K = 0.001
    direction_vector = to_vector(pusher, pushee)
    force_magnitude = K*CHARGE1*CHARGE2/get_length(direction_vector)**2
    return scale(direction_vector, force_magnitude)


def generate_random_unit_vectors(m, n):
    ''' generates m n-dimensional normal vectors '''
    return [normalize(get_random_vector(n)) for _ in range(m)]

def spread_vectors(vectors, number_of_iterations=50):
    '''spreads vectors evenly on an n-dimensional hypersphere '''

    for iteration in range(number_of_iterations):

        snapshot = deepcopy(vectors)  # preserve the past to build the future
        for index, old_vector in enumerate(snapshot):
            total_force = create_zero_vector(len(old_vector))
            for other_vector in list_without(snapshot, index):
                force = get_coulombs_force(pusher=other_vector, pushee=old_vector)
                total_force = add(total_force, force)
                force = get_coulombs_force(pusher=reflect(other_vector), pushee=old_vector)
                total_force = add(total_force, force)
            vectors[index] = normalize(add(old_vector, total_force))
    return vectors


def plot_3d_vectors(vectors, title='vectors'):
    x, y, z = zip(*vectors)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'ro')
    plt.axis([-2, 2, -2, 2])
    plt.show()

def plot_2d_vectors(vectors, title='vectors'):
    x, y = zip(*vectors)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'ro')
    plt.axis([-2, 2, -2, 2])
    plt.show()


def main():
    vectors = generate_random_unit_vectors(17, 2)
    plot_2d_vectors(vectors, title='Before spread')
    vectors = spread_vectors(vectors, number_of_iterations=500)
    plot_2d_vectors(vectors, title='After spread')
    # pusher = [0.3, 0.2]
    # pushee = [0.6, 0.1]
    # force = get_coulombs_force(pusher, pushee)
    # print(force)
    # plot_vectors([pusher, pushee, force, [0,0]])

    # print(to_hash_vector(file_to_words('article'), NUMBER_OF_VECTORS))

if __name__ == '__main__':
    logging.basicConfig(filename=LOGFILE, level=logging.INFO)
    log = logging.getLogger(__name__)
    set_random_seed(RANDOM_SEED)
    main()
