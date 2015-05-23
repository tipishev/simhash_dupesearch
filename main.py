#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, absolute_import

__author__ = 'tipishev'

from os import walk

LOGFILE = "main.log"

import logging
logging.basicConfig(filename=LOGFILE, level=logging.INFO)
log = logging.getLogger(__name__)


def count_words(source_directory, output_file):
    dictionary = dict()
    for (dirpath, dirnames, filenames) in walk(source_directory):
        total_length = len(filenames)
        for count, filename in enumerate(filenames):
            fullpath = dirpath+filename
            with open(fullpath, 'r') as f:
                for word in f.read().split():
                    if word not in dictionary:
                        dictionary[word] = 1
                    else:
                        dictionary[word] += 1
                print("Writing {} of {}".format(count, total_length))

    with codecs.open(output_file, 'w', 'utf8') as g:
        for (key, value) in dictionary.iteritems():
            # g.write('35')
            g.write("{} {}\n".format(key.decode('utf8'), value))

    # with codecs.open(target_directory+filename, 'w', 'utf8') as g:
    #     g.write(text)

def main():
    TARGET_DIRECTORY = "../docs/"

    count_words(TARGET_DIRECTORY, 'count')

if __name__ == '__main__':
    main()
