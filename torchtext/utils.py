import six
import requests
import csv
import os
import tarfile
import logging


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner


def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive

    Arguments:
        url: the url for online Dataset
        path: directory and filename for the downloaded dataset.

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> path = './validation.tar.gz'
        >>> torchtext.utils.download_from_url(url, path)
    """

    logger = logging.getLogger('torchtext')

    def process_response(r):
        chunk_size = 16 * 1024
        with open(path, "wb") as file:
            downloaded = 0
            for chunk in r.iter_content(chunk_size):
                if chunk:
                    file.write(chunk)
                    if downloaded % (chunk_size * 16) == 0:
                        logger.info('Downloaded {} bytes'.format(downloaded))
                    downloaded += len(chunk)
            logger.info('Downloaded {} bytes'.format(downloaded))

    if 'drive.google.com' not in url:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        process_response(response)
        return path

    logger.info('Downloading to path ' + path)
    logger.info('Downloading from Google Drive')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    process_response(response)
    return path


def unicode_csv_reader(unicode_csv_data, **kwargs):
    """Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples"""
    if six.PY2:
        # csv.py doesn't do Unicode; encode temporarily as UTF-8:
        csv_reader = csv.reader(utf_8_encoder(unicode_csv_data), **kwargs)
        for row in csv_reader:
            # decode UTF-8 back to Unicode, cell by cell:
            yield [cell.decode('utf-8') for cell in row]
    else:
        for line in csv.reader(unicode_csv_data, **kwargs):
            yield line


def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')


def extract_archive(from_path, to_path=None):
    """Extract tar.gz archives.

    Arguments:
        from_path: the path where the tar.gz file is.
        to_path: the path where the extracted files are.

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)
    logger = logging.getLogger('torchtext')
    logger.info('Extracting {} to path {}'.format(from_path, to_path))

    # We rely on tarfile to throw an exception if the given file isn't
    # a valid archive
    with tarfile.open(from_path, 'r:gz') as tar:
        tar.extractall(path=to_path)

    return to_path
