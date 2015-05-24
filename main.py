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
from copy import deepcopy
from cPickle import dump, load
from itertools import combinations


LOGFILE = "main.log"
DOCS_DIRECTORY = "./docs/"
VECTORS_FILE = 'vectors.pkl'
DISTANCES_FILE = 'distances.pkl'
DOCSTATS_FILE = 'docstats.pkl'
SORTED_FILE = 'sorted.pkl'


RANDOM_SEED = 42
NUMBER_OF_DIMENSIONS = 42  # a.k.a. N
NUMBER_OF_VECTORS = 64  # a.k.a. M


# silly utils

def list_without(list_, k):
    ''' return a list without element at position k '''
    return list_[:k] + list_[(k+1):]


# string stuff

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


def to_simhash(doc_hash, vectors):
    return ''.join(['1' if scalar(doc_hash, v) > 0 else '0' for v in vectors])


# vector stuff


def create_zero_vector(size):
    return [0]*size


def normalize(vector):
    return(scale(vector, 1/get_length(vector)))


def to_vector(point1, point2):
    assert len(point1) == len(point2)
    return [e2-e1 for e1, e2 in zip(point1, point2)]


def add(vector1, vector2):
    return [sum(e) for e in zip(vector1, vector2)]


def scalar(vector1, vector2):
    return sum([e[0]*e[1] for e in zip(vector1, vector2)])


def scale(vector, k):
    return [k*e for e in vector]


def get_length(vector):
    return sqrt(sum([e*e for e in vector]))


def reflect(vector):
    return [-e for e in vector]


# vector randomization magic


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
    '''spreads vectors evenly on an n-dimensional sphere '''

    for iteration in range(number_of_iterations):
        log.info("iteration #{} of {}".format(iteration, number_of_iterations))

        snapshot = deepcopy(vectors)  # preserve the past to build the future
        for index, old_vector in enumerate(snapshot):
            total_force = create_zero_vector(len(old_vector))
            for other_vector in list_without(snapshot, index):
                force = get_coulombs_force(pusher=other_vector,
                                           pushee=old_vector)
                total_force = add(total_force, force)
                force = get_coulombs_force(pusher=reflect(other_vector),
                                           pushee=old_vector)
                total_force = add(total_force, force)
            vectors[index] = normalize(add(old_vector, total_force))
    return vectors


# def plot_3d_vectors(vectors, title='vectors'):
#     from mpl_toolkits.mplot3d import Axes3D
#     x, y, z = zip(*vectors)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x, y, z)
#     plt.show()


def plot_2d_vectors(vectors, title='vectors'):
    x, y = zip(*vectors)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'ro')
    plt.axis([-2, 2, -2, 2])
    plt.show()


def dump_vectors(m=NUMBER_OF_VECTORS, n=NUMBER_OF_DIMENSIONS,
                 filename=VECTORS_FILE):
    ''' generate m well-spread n-dimensional vectors and dump 'em to a file '''
    vectors = generate_random_unit_vectors(m=m, n=n)
    vectors = spread_vectors(vectors, number_of_iterations=100)
    with open(filename, 'wb') as f:
        dump(vectors, f)
    log.info('saved the vectors')


def load_vectors(filename=VECTORS_FILE):
    with open(filename, 'rb') as f:
        log.info('loaded the vectors')
        return load(f)


def dump_distances(distances, filename=DISTANCES_FILE):
    ''' save a dict with hamming distances between files (poor man's hash) '''
    with open(filename, 'wb') as f:
        dump(distances, f)
    log.info('saved the distances')


def load_distances(filename=DISTANCES_FILE):
    with open(filename, 'rb') as f:
        log.info('loaded the distances')
        return load(f)


def dump_docstats(docstats, filename=DOCSTATS_FILE):
    ''' save a dict with docstats for each file '''
    with open(filename, 'wb') as f:
        dump(docstats, f)
    log.info('saved the docstats')


def load_docstats(filename=DOCSTATS_FILE):
    with open(filename, 'rb') as f:
        log.info('loaded the docstats')
        return load(f)


def dump_sorted(sorted_array, filename=SORTED_FILE):
    with open(filename, 'wb') as f:
        dump(sorted_array, f)
    log.info('saved the sorted array')


def load_sorted(filename=SORTED_FILE):
    with open(filename, 'rb') as f:
        log.info('loaded the sorted array')
        return load(f)


def handle_file(filename, vectors):
    words = file_to_words(filename)
    hash_vector = to_hash_vector(words=words, length=len(vectors))
    return {'wordcount': len(words),
            'simhash': to_simhash(hash_vector, vectors)}


def collect_doc_stats(directory):
    vectors = load_vectors()
    stats = dict()
    for (dirpath, dirnames, filenames) in walk(directory):
        filecount = len(filenames)
        for count, filename in enumerate(filenames):
            log.info("processing #{} of {} files".format(count, filecount))
            fullpath = dirpath+filename
            stats[fullpath] = handle_file(fullpath, vectors)
    return stats


def relative_change(old, new):
    return float(new-old)/old


def generate_windows(array, max_gap=0.20):
    ''' yields lists of indices, whose values in the array are within the gap '''
    min_index = min_value = None
    window = []
    for index, value in enumerate(array):
        if min_index is None:
            min_index, min_value = index, value
            window.append(index)
        diff = relative_change(min_value, value)
        if diff > max_gap:
            yield window
            min_index += 1
            min_value = array[min_index]
            window = [index]
        else:
            window.append(index)
    yield window


def get_window_sizes(array, max_gap=0.20, filter_ones=False):
    return sorted([len(w) for w in generate_windows(array, max_gap) if len(w) > 1 or not filter_ones])


def get_number_of_comparisons(sizes):
    return [s*(s-1)/2 for s in sizes]

def calculate_distances(docstats):
    wordcounts = [e[0] for e in docstats]
    distances = dict()
    for indices_window in generate_windows(wordcounts):
        if len(indices_window) > 1:
            to_compare = combinations(indices_window, 2)
            for pair in to_compare:
                # filename1 = docstats[pair[0]][1]['filename']
                # filename2 = docstats[pair[1]][1]['filename']
                # print('will compare {} and {}'.format(filename1, filename2))
                simhash1 = docstats[pair[0]][1]['simhash']
                simhash2 = docstats[pair[1]][1]['simhash']
                distances[pair] = hamming_distance(simhash1, simhash2)
                print(len(distances))
    return distances

def estimate_complexity(docstats, max_gap=0.2):
    wordcounts = [e[0] for e in docstats]
    sizes = get_window_sizes(wordcounts, max_gap=max_gap, filter_ones=True)
    print("the nontrivial comparison window sizes are: \n {}\n".format(sizes))
    print(("the total number of comparisons required is {}, "
           "which is less than the total number of atoms in the universe").format(
               sum(get_number_of_comparisons(sizes))))



def main():
    # load a precalculated and sorted array of (wordcount: {'simhash', 'filename'})
    docstats = load_sorted()
    # estimate_complexity(docstats)
    # distances = calculate_distances(docstats)
    # dump_distances(distances)
    distances = load_distances()
    counter = 0
    for (index1, index2), distance in distances.iteritems():
        print (docstats[index1][1]['filename'], docstats[index2][1]['filename'], distance)
        counter += 1
        if counter > 32:
            break


if __name__ == '__main__':
    logging.basicConfig(filename=LOGFILE, level=logging.INFO)
    log = logging.getLogger(__name__)
    set_random_seed(RANDOM_SEED)
    main()
