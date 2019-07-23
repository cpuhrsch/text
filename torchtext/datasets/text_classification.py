import os
import re
import logging
import torch
from torchtext.utils import download_from_url, extract_archive
from torchtext import data
from torchtext.vocab import build_dictionary
import random
from tqdm import tqdm


def download_extract_archive(url, raw_folder, dataset_name):
    """Download the dataset if it doesn't exist in processed_folder already."""

    train_csv_path = os.path.join(raw_folder,
                                  dataset_name + '_csv',
                                  'train.csv')
    test_csv_path = os.path.join(raw_folder,
                                 dataset_name + '_csv',
                                 'test.csv')
    if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
        return

    os.makedirs(raw_folder)
    filename = dataset_name + '_csv.tar.gz'
    url = url
    path = os.path.join(raw_folder, filename)
    download_from_url(url, path)
    extract_archive(path, raw_folder, remove_finished=True)

    logging.info('Dataset %s downloaded.' % dataset_name)

# TODO: Replicate below
#  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
#    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
#        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
#        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " "
_normalize_pattern_re = re.compile(r'[\W_]+')

def text_normalize(line):
    """
    Basic normalization for a line of text.

    Normalization includes
    - lowercasing
    - replacing all non-alphanumeric characters with whitespace

    Returns a list of tokens after splitting on whitespace.
    """

    line = line.lower()
    line = _normalize_pattern_re.sub(' ', line)


    return line.split()

URLS = {
    'AG_NEWS':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms',
    'SogouNews':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE',
    'DBpedia':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k',
    'YelpReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg',
    'YelpReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0',
    'YahooAnswers':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU',
    'AmazonReviewPolarity':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM',
    'AmazonReviewFull':
        'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'
}
