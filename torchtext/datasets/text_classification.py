import os
import re
import torch
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import generate_ngrams
from torchtext.vocab import build_vocab
import random
from torchtext import data
from tqdm import tqdm
from torchtext.data import dataset
from torchtext.data.iterator import generate_iterators


def download(url, raw_folder, dataset_name):
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

    print('Dataset %s downloaded.' % dataset_name)


def text_normalize(line):
    """Normalize text string and separate label/text."""

    line = line.lower()
    label, text = line.split(",", 1)
    label = "__label__" + re.sub(r'[^0-9\s]', '', label)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    line = label + ' , ' + text + ' \n'
    return line


def preprocess(raw_folder, processed_folder, dataset_name):
    """Preprocess the csv files."""

    raw_folder = os.path.join(raw_folder, dataset_name.lower() + '_csv')

    if os.path.exists(processed_folder) is not True:
        os.makedirs(processed_folder)

    src_filepath = os.path.join(raw_folder, 'train.csv')
    tgt_filepath = os.path.join(processed_folder, dataset_name + '.train')
    lines = []
    with open(src_filepath) as src_data:
        with open(tgt_filepath, 'w') as new_data:
            for line in src_data:
                lines.append(text_normalize(line))
            random.shuffle(lines)
            new_data.writelines(lines)

    src_filepath = os.path.join(raw_folder, 'test.csv')
    tgt_filepath = os.path.join(processed_folder, dataset_name + '.test')
    lines = []
    with open(src_filepath) as src_data:
        with open(tgt_filepath, 'w') as new_data:
            for line in src_data:
                lines.append(text_normalize(line))
            random.shuffle(lines)
            new_data.writelines(lines)

    print("Dataset %s preprocessed." % dataset_name)


def load_text_classification_data(filepath, fields, ngrams=1):
    """Load train/test data from a file and generate data
        examples with ngrams.
    """

    def label_text_processor(line, fields, ngrams=1):
        """Process text string and generate examples for dataset."""
        fields = [('text', fields['text']), ('label', fields['label'])]
        label, text = line.split(",", 1)
        label = float(label.split("__label__")[1])
        ex = data.Example.fromlist([text, label], fields)
        tokens = ex.text[1:]  # Skip the first space '\t'
        ex.text = generate_ngrams(tokens, ngrams)
        return ex

    examples = []
    with open(filepath) as src_data:
        for line in tqdm(src_data):
            examples.append(label_text_processor(line, fields, ngrams))
    return examples


def generate_iters(train_examples, test_examples, fields, sort_key,
          split_ratio=0.7, batch_size=32, device='cpu', random_state=None):
    """Create iterator objects for splits of the dataset.

    Arguments:
        split_ratio: split train_examples into train set (split_ratio)
            and valid set (1-split_ratio). Default: 0.7
        batch_size: batch size. Default: 32
        device: the device to sent data. Default: 'cpu'
        random_state: the random state provided by user. Default: None

    Examples:
        >>> train_iter, test_iter, valid_iter = txt_cls.generate_iters(device="cpu")

    Outputs:
        - train_iter: a iterator based on
            train.csv file with split_ratio percent.
        - train_iter: a iterator based on test.csv file.
        - valid_iter: a iterator based on
            train.csv file with 1-split_ratio percent.

    """

    rnd = dataset.RandomShuffler(random_state)
    train_ratio, valid_ratio = split_ratio, 1 - split_ratio
    train_examples, _test_examples, valid_examples = \
        dataset.rationed_split(train_examples, train_ratio, 0.0,
                               valid_ratio, rnd)
    train = TextDataset(train_examples, fields)
    train.sort_key = sort_key
    test = TextDataset(test_examples, fields)
    test.sort_key = sort_key
    valid = TextDataset(valid_examples, fields)
    valid.sort_key = sort_key

    return generate_iterators(
        (train, test, valid), [batch_size] * 3, device=device)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = fields

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.examples:
            yield x


class TextClassificationDataset(TextDataset):
    """Defines text classification datasets.
        Currently, we only support the following datasets:

             - AG_NEWS
             - SogouNews
             - DBpedia
             - YelpReviewPolarity
             - YelpReviewFull
             - YahooAnswers
             - AmazonReviewPolarity
             - AmazonReviewFull

    """

    def __init__(self, url, root='.data',
                 text_field=None, label_field=None, ngrams=1):
        """Create a text classification dataset instance.

        Arguments:
            dataset_name: The name of dataset, include "ag_news", "sogou_news",
                "dbpedia", "yelp_review_polarity", "yelp_review_full",
                "yahoo_answers", "amazon_review_full", "amazon_review_polarity".
            root: Directory where the dataset are saved. Default: ".data"
            text_field: The field that will be used for the sentence. If not given,
                'spacy' token will be used.
            label_field: The field that will be used for the label. If not given,
                'float' token will be used.
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1

        Examples:
            >>> url = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms'
            >>> txt_cls = TextClassificationDataset(url, ngrams=2)

        """
        fields = []
        fields.append(('text', text_field if text_field is not None
                       else data.Field(tokenize=data.get_tokenizer('spacy'),
                                       init_token='<SOS>',
                                       eos_token='<EOS>')))
        fields.append(('label', label_field if label_field is not None
                       else data.LabelField(dtype=torch.float)))

        super(TextClassificationDataset, self).__init__([], dict(fields))

        self.dataset_name = self.__class__.__name__
        self.root = root
        self.raw_folder = os.path.join(self.root, self.__class__.__name__, 'raw')
        self.processed_folder = os.path.join(self.root,
                                             self.__class__.__name__,
                                             'processed')
        filepath = os.path.join(self.processed_folder, self.dataset_name + '.train')
        if not os.path.isfile(filepath):
            download(self.url, self.raw_folder, self.dataset_name)
            preprocess(self.raw_folder, self.processed_folder, self.dataset_name)
        self.train_examples = load_text_classification_data(filepath, self.fields, ngrams)
        filepath = os.path.join(self.processed_folder, self.dataset_name + '.test')
        self.test_examples = load_text_classification_data(filepath, self.fields, ngrams)
        self.examples = self.train_examples + self.test_examples
        self.fields['text'].vocab = build_vocab(self, self.fields['text'], 'text')
        self.fields['label'].vocab = build_vocab(self, self.fields['label'], 'label')

    @staticmethod
    def sort_key(ex):
        return len(ex.text)


class AG_NEWS(TextClassificationDataset):
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 1 : World
            - 2 : Sports
            - 3 : Business
            - 4 : Sci/Tech
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: AG_NEWS

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.AG_NEWS(ngrams=3)

        """
        self.url = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms'
        super(AG_NEWS, self).__init__(self.url, root, text_field,
                                      label_field, ngrams)


class SogouNews(TextClassificationDataset):
    """ Defines SogouNews datasets.
        The labels includes:
            - 1 : Sports
            - 2 : Finance
            - 3 : Entertainment
            - 4 : Automobile
            - 5 : Technology
     """
    def __init__(self, root='.data', text_field=None,
                 label_field=None, ngrams=1):
        """Create supervised learning dataset: SogouNews

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.SogouNews(ngrams=3)

        """
        self.url = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE'
        super(SogouNews, self).__init__(self.url, root, text_field,
                                        label_field, ngrams)


class DBpedia(TextClassificationDataset):
    """ Defines DBpedia datasets.
        The labels includes:
            - 1 : Company
            - 2 : EducationalInstitution
            - 3 : Artist
            - 4 : Athlete
            - 5 : OfficeHolder
            - 6 : MeanOfTransportation
            - 7 : Building
            - 8 : NaturalPlace
            - 9 : Village
            - 10 : Animal
            - 11 : Plant
            - 12 : Album
            - 13 : Film
            - 14 : WrittenWork
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: DBpedia

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.DBpedia(ngrams=3)

        """
        self.url = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k'
        super(DBpedia, self).__init__(self.url, root, text_field, label_field, ngrams)


class YelpReviewPolarity(TextClassificationDataset):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: YelpReviewPolarity

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.YelpReviewPolarity(ngrams=3)

        """
        self.url = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg'
        super(YelpReviewPolarity, self).__init__(self.url,
                                                 root, text_field, label_field, ngrams)


class YelpReviewFull(TextClassificationDataset):
    """ Defines YelpReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended).
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: YelpReviewFull

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.YelpReviewFull(ngrams=3)

        """
        self.url = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0'
        super(YelpReviewFull, self).__init__(self.url,
                                             root, text_field, label_field, ngrams)


class YahooAnswers(TextClassificationDataset):
    """ Defines YahooAnswers datasets.
        The labels includes:
            - 1 : Society & Culture
            - 2 : Science & Mathematics
            - 3 : Health
            - 4 : Education & Reference
            - 5 : Computers & Internet
            - 6 : Sports
            - 7 : Business & Finance
            - 8 : Entertainment & Music
            - 9 : Family & Relationships
            - 10 : Politics & Government
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: YahooAnswers

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.YahooAnswers(ngrams=3)

        """
        self.url = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU'
        super(YahooAnswers, self).__init__(self.url,
                                           root, text_field, label_field, ngrams)


class AmazonReviewPolarity(TextClassificationDataset):
    """ Defines AmazonReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity
            - 2 : Positive polarity
     """
    def __init__(self, root='.data', text_field=None,
                 label_field=None, ngrams=1):
        """Create supervised learning dataset: AmazonReviewPolarity

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.AmazonReviewPolarity(ngrams=3)

        """
        self.url = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM'
        super(AmazonReviewPolarity, self).__init__(self.url,
                                                   root, text_field, label_field, ngrams)


class AmazonReviewFull(TextClassificationDataset):
    """ Defines AmazonReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended)
     """
    def __init__(self, root='.data', text_field=None, label_field=None, ngrams=1):
        """Create supervised learning dataset: AmazonReviewFull

        Inputs:
            See TextClassificationDataset() class

        Examples:
            >>> text_cls = torchtext.datasets.AmazonReviewFull(ngrams=3)

        """
        self.url = 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA'
        super(AmazonReviewFull, self).__init__(self.url,
                                               root, text_field, label_field, ngrams)
