import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.datasets.raw import language_modeling as raw
from torchtext.experimental.functional import vocab_func, totensor, sequential_transforms
from torch.utils.data import DataLoader


def build_vocab(data, transforms):
    def apply_transforms(data):
        for line in data:
            tokens = transforms(line)
            if len(tokens) > 0:
                yield tokens
    return build_vocab_from_iterator(apply_transforms(data))


class LanguageModelingDataset(torch.utils.data.Dataset):
    """Defines a dataset for language modeling.
       Currently, we only support the following datasets:

             - WikiText2
             - WikiText103
             - PennTreebank
             - WMTNewsCrawl

    """

    def __init__(self, data, vocab, transforms, single_line):
        """Initiate language modeling dataset.

        Arguments:
            data: a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab: Vocabulary object used for dataset.
            transforms: Text string transforms.

        """

        super(LanguageModelingDataset, self).__init__()
        self.vocab = vocab
        self.transforms = transforms
        self.single_line = single_line
        self.data = data
        # if single_line:
        #     tmp_data = []
        #     for d in self.data:
        #         if d.numel() > 0:
        #             tmp_data.append(d[0])
        #     self.data = torch.cat(tmp_data)

    def __getitem__(self, i):
        if self.single_line:
            return self.data[i]
        else:
            return self.transforms(self.data[i])

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab


class _IterableWrapper(torch.utils.data.IterableDataset):
    def __init__(self, iterator, num_lines, tokenizer, vocab):
        super(_IterableWrapper, self).__init__()
        self.num_lines = num_lines
        self.init_iterator = iterator
        self.iterator = None
        self.tokenizer = tokenizer
        self.vocab = vocab

    def setup_iterator(self, init_iterator):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            chunk = int(self.num_lines / worker_info.num_workers)
            start = chunk * worker_info.id
            read = chunk
            if worker_info.id == worker_info.num_workers - 1:
                # The last worker needs to pick up some extra lines
                # if the number of lines aren't exactly divisible
                # by the number of workers.
                extra = self.num_lines % worker_info.num_workers
                read += extra
        else:
            start = 0
            read = self.num_lines
        return init_iterator(start, read)

    def __iter__(self):
        if not self.iterator:
            self.iterator = self.setup_iterator(self.init_iterator)
        for d in self.iterator:
            yield d
            # yield torch.tensor([self.vocab[t] for t in self.tokenizer(d)], dtype=torch.long)


def _setup_datasets(dataset_name, tokenizer=None, root='.data', vocab=None,
                    data_select=('train', 'test', 'valid'), single_line=True):
    if tokenizer is None:
        tokenizer = get_tokenizer('basic_english')
    text_transform = sequential_transforms(tokenizer)

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'valid', 'test'))):
        raise TypeError('Given data selection {} is not supported!'.format(data_select))

    if not single_line and dataset_name != 'WikiText103':
        raise TypeError('single_line must be True except for WikiText103')
    if vocab is None:
        if 'train' not in data_select:
            raise TypeError("Must pass a vocab if train is not selected.")
        train, = raw.DATASETS[dataset_name](root=root, data_select=('train',))
        vocab = build_vocab(train, text_transform)

    # text_transform = sequential_transforms(tokenizer, vocab_func(vocab), totensor(dtype=torch.long))
    raw_iter = raw.DATASETS[dataset_name](root=root, data_select=data_select)
    raw_num_lines = {}
    for i in range(len(raw_iter)):
        name = data_select[i]
        raw_num_lines[name] = sum(1 for d in raw_iter[i])

    raw_data = {}
    for i in range(len(raw_iter)):
        name = data_select[i]

        def build_raw_iter(start, num_lines):
            raw_iter, = raw.DATASETS[dataset_name](root=root, data_select=(name,), start=start, num_lines=num_lines)
            return raw_iter
        num_lines = raw_num_lines[name]
        # DataLoader overhead is high enough that we need to know its worth to construct
        if raw_num_lines[name] < 1:  # 100000:
            def text_transform(line):
                tokens = tokenizer(line)
                return torch.tensor([vocab[t] for t in tokens], dtype=torch.long)
            raw_data[name] = [text_transform(txt) for txt in build_raw_iter(0, num_lines)]
        else:
            data_iter = DataLoader(_IterableWrapper(build_raw_iter, num_lines, tokenizer, vocab.stoi),
                                   num_workers=torch.get_num_threads())
            raw_data[name] = []
            for txt in data_iter:
                raw_data[name].append(txt)
            del data_iter
    return tuple(LanguageModelingDataset(raw_data[item], vocab, lambda x: x, single_line)
                 for item in data_select)


def WikiText2(*args, **kwargs):
    """ Defines WikiText2 datasets.

    Create language modeling dataset: WikiText2
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        single_line: whether to return all tokens in a single line.
            (Default: True)
            By default, all lines in raw text file are concatenated into a single line.
            Use `single_line = False` if one wants to get data line by line.

    Examples:
        >>> from torchtext.experimental.datasets import WikiText2
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText2(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText2(tokenizer=tokenizer, vocab=vocab,
                                       data_select='valid')

    """

    return _setup_datasets(*(("WikiText2",) + args), **kwargs)


def WikiText103(*args, **kwargs):
    """ Defines WikiText103 datasets.

    Create language modeling dataset: WikiText103
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        single_line: whether to return all tokens in a single line.
            (Default: True)
            By default, all lines in raw text file are concatenated into a single line.
            Use `single_line = False` if one wants to get data line by line.

    Examples:
        >>> from torchtext.experimental.datasets import WikiText103
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = WikiText103(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = WikiText103(tokenizer=tokenizer, vocab=vocab,
                                         data_select='valid')

    """

    return _setup_datasets(*(("WikiText103",) + args), **kwargs)


def PennTreebank(*args, **kwargs):
    """ Defines PennTreebank datasets.

    Create language modeling dataset: PennTreebank
    Separately returns the train/test/valid set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
        single_line: whether to return all tokens in a single line.
            (Default: True)
            By default, all lines in raw text file are concatenated into a single line.
            Use `single_line = False` if one wants to get data line by line.

    Examples:
        >>> from torchtext.experimental.datasets import PennTreebank
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, test_dataset, valid_dataset = PennTreebank(tokenizer=tokenizer)
        >>> vocab = train_dataset.get_vocab()
        >>> valid_dataset, = PennTreebank(tokenizer=tokenizer, vocab=vocab,
                                          data_select='valid')

    """

    return _setup_datasets(*(("PennTreebank",) + args), **kwargs)


def WMTNewsCrawl(*args, **kwargs):
    """ Defines WMTNewsCrawl datasets.

    Create language modeling dataset: WMTNewsCrawl
    returns the train set

    Arguments:
        tokenizer: the tokenizer used to preprocess raw text data.
            The default one is basic_english tokenizer in fastText. spacy tokenizer
            is supported as well (see example below). A custom tokenizer is callable
            function with input of a string and output of a token list.
        root: Directory where the datasets are saved. Default: ".data"
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        data_select: a string or tupel for the returned datasets
            (Default: ('train',))
        single_line: whether to return all tokens in a single line.
            (Default: True)
            By default, all lines in raw text file are concatenated into a single line.
            Use `single_line = False` if one wants to get data line by line.
    Examples:
        >>> from torchtext.experimental.datasets import WMTNewsCrawl
        >>> from torchtext.data.utils import get_tokenizer
        >>> tokenizer = get_tokenizer("spacy")
        >>> train_dataset, = WMTNewsCrawl(tokenizer=tokenizer, data_select='train')

    """

    return _setup_datasets(*(("WMTNewsCrawl",) + args), **kwargs)


DATASETS = {
    'WikiText2': WikiText2,
    'WikiText103': WikiText103,
    'PennTreebank': PennTreebank,
    'WMTNewsCrawl': WMTNewsCrawl
}
