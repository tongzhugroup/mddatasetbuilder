from multiprocessing import Pool, Semaphore
import os
import logging
import itertools
import pickle

from tqdm import tqdm
import lz4.frame
import pybase64


def multiopen(pool, func, l, semaphore=None, nlines=None, unordered=True, return_num=False, start=0, extra=None, interval=None, bar=True, desc=None, unit="it", total=None):
    obj = l
    if nlines:
        obj = itertools.zip_longest(*[obj] * nlines)
    if interval:
        obj = itertools.islice(obj, 0, None, interval)
    if return_num:
        obj = enumerate(obj, start)
    if semaphore:
        obj = produce(semaphore, obj, extra)
    if unordered:
        obj = pool.imap_unordered(func, obj, 100)
    else:
        obj = pool.imap(func, obj, 100)
    if bar:
        obj = tqdm(obj, desc=desc, unit=unit, total=total)
    return obj


def produce(semaphore, plist, parameter):
    """Prevent large memory usage due to slow IO."""
    for item in plist:
        semaphore.acquire()
        if parameter is not None:
            item = (item, parameter)
        yield item


def compress(x, isbytes=False):
    """Compress the line.

    This function reduces IO overhead to speed up the program.
    """
    if isbytes:
        return pybase64.b64encode(lz4.frame.compress(x, compression_level=0))+b'\n'
    return pybase64.b64encode(lz4.frame.compress(x.encode(), compression_level=-1))+b'\n'


def decompress(x, isbytes=False):
    """Decompress the line."""
    if isbytes:
        return lz4.frame.decompress(pybase64.b64decode(x.strip(), validate=True))
    return lz4.frame.decompress(pybase64.b64decode(x.strip(), validate=True)).decode()


def listtobytes(x):
    return compress(pickle.dumps(x), isbytes=True)


def bytestolist(x):
    return pickle.loads(decompress(x, isbytes=True))


def run_mp(nproc, **arg):
    pool = Pool(nproc, maxtasksperchild=1000)
    semaphore = Semaphore(nproc*150)
    try:
        results = multiopen(pool=pool, semaphore=semaphore, **arg)
        for item in results:
            yield item
            semaphore.release()
    except:
        logging.exception("run_mp failed")
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()


def must_be_list(obj):
    if isinstance(obj, list):
        return obj
    return [obj]


def _mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass
