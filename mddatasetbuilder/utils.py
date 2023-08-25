import itertools
import pickle
from multiprocessing import Pool, Semaphore
from typing import BinaryIO, Union

import lz4.frame
from tqdm.auto import tqdm

from ._logger import logger


def multiopen(
    pool,
    func,
    l,
    semaphore=None,
    nlines=None,
    unordered=True,
    return_num=False,
    start=0,
    extra=None,
    interval=None,
    bar=True,
    desc=None,
    unit="it",
    total=None,
):
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
        obj = tqdm(obj, desc=desc, unit=unit, total=total, disable=None)
    return obj


def produce(semaphore, plist, parameter):
    """Prevent large memory usage due to slow IO."""
    for item in plist:
        semaphore.acquire()
        if parameter is not None:
            item = (item, parameter)
        yield item


def compress(x: Union[str, bytes]) -> bytes:
    """Compress the line.

    This function reduces IO overhead to speed up the program. The functions will
    use lz4 to compress, since lz4 has better performance that any others.
    The compressed format is size + data + size + data + ..., where size is a 64-bit
    little-endian integer.

    Parameters
    ----------
    x : str or bytes
        The line to compress.

    Returns
    -------
    bytes
        The compressed line, with a linebreak in the end.
    """
    if isinstance(x, str):
        x = x.encode()
    compress_block = lz4.frame.compress(x, compression_level=0)
    length_bytes = len(compress_block).to_bytes(64, byteorder="little")
    return length_bytes + compress_block


def decompress(x: bytes, isbytes: bool = False) -> Union[str, bytes]:
    """Decompress the line.

    Parameters
    ----------
    x : bytes
        The line to decompress.
    isbytes : bool, optional, default: False
        If the decompressed content is bytes. If not, the line will be decoded.

    Returns
    -------
    str or bytes
        The decompressed line.
    """
    x = lz4.frame.decompress(x[64:])
    if isbytes:
        return x
    return x.decode()


def read_compressed_block(f: BinaryIO):
    """Read compressed binary file, assuming the format is size + data + size + data + ...

    Parameters
    ----------
    f : fileObject
        The file object to read.

    Yields
    ------
    data: bytes
        The compressed block.
    """
    while True:
        sizeb = f.read(64)
        if not sizeb:
            break
        size = int.from_bytes(sizeb, byteorder="little")
        yield sizeb + f.read(size)


def listtobytes(x):
    return compress(pickle.dumps(x))


def bytestolist(x):
    return pickle.loads(decompress(x, isbytes=True))


def run_mp(nproc, **arg):
    pool = Pool(nproc, maxtasksperchild=1000)
    semaphore = Semaphore(nproc * 150)
    try:
        results = multiopen(pool=pool, semaphore=semaphore, **arg)
        for item in results:
            yield item
            semaphore.release()
    except:
        logger.exception("run_mp failed")
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
