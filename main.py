#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, absolute_import

__author__ = 'tipishev'

from os import walk
# from md5 import digest as md5sum
import md5
from string import punctuation
from re import sub

# filesystem stuff
LOGFILE = "main.log"
DOCS_DIRECTORY = "../docs/"

# algorithm stuff
NUMBER_OF_DIMENSIONS = 64
NUMBER_OF_VECTORS = 5

import logging
logging.basicConfig(filename=LOGFILE, level=logging.INFO)
log = logging.getLogger(__name__)


def md5sum(string):
    return md5.new(string).hexdigest()

def distance(first, second):
    assert len(first)==len(second)
    return sum([a!=b for (a, b) in zip(first, second)])

def to_words(raw_text):
    return sub('[{}]'.format(punctuation), ' ', raw_text.lower()).split()

def to_text(filename):
    with open(filename, 'r') as f:
        return ' '.join(line for line in f)

def to_binary(hexstring):
    return bin(int(hexstring, 16))[2:]

def to_hash_sum_vector(filename, length):
    ''' create a sum vector depending on each word's md5 sum '''
    result = [0 for _ in range(length)]
    for word in to_words((to_text(filename))):
        control_vector = to_binary(md5sum(word))[0:length]
        for position, char in enumerate(control_vector):
            result[position] += 1 if char == '1' else -1
    return result


def main():
    print(to_hash_sum_vector('34567', NUMBER_OF_VECTORS))

if __name__ == '__main__':
    main()
