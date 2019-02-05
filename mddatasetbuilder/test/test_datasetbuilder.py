"""Test.

python setup.py pytest
"""


import json
import logging
import math
import os
import hashlib
import tempfile

import pkg_resources
import requests
from tqdm import tqdm

import mddatasetbuilder


class TestMDDatasetBuilder:
    """Test MDDatasetBuilder."""

    def test_datasetbuilder(self):
        """Test DatasetBuilder."""
        folder = tempfile.mkdtemp(prefix='testfiles-', dir='.')
        logging.info(f'Folder: {folder}:')
        os.chdir(folder)
        testparms = json.load(
            pkg_resources.resource_stream(__name__, 'test.json'))
        # download bonds.reaxc and dump.reaxc
        for fileparms in (testparms["bondfile"], testparms["dumpfile"]):
            self._download_file(fileparms["url"],
                                fileparms["filename"],
                                fileparms["sha256"])

        d = mddatasetbuilder.DatasetBuilder(
            bondfilename=testparms["bondfile"]["filename"],
            dumpfilename=testparms["dumpfile"]["filename"],
            atomname=testparms["atomname"],
            dataset_name=testparms["dataset_name"],
            stepinterval=testparms["stepinterval"]
            if "stepinterval" in testparms else 1)
        d.builddataset()

        assert os.path.exists(d.gjfdir)

    def _download_file(self, urls, pathfilename, sha256):
        times = 0
        # download if not exists
        while times < 3:
            if os.path.isfile(pathfilename) and self._checksha256(
                    pathfilename, sha256):
                break
            try:
                os.makedirs(os.path.split(pathfilename)[0])
            except OSError:
                pass

            # from https://stackoverflow.com/questions/16694907
            if not isinstance(urls, list):
                urls = [urls]
            for url in urls:
                try:
                    logging.info(f"Try to download {pathfilename} from {url}")
                    r = requests.get(url, stream=True)
                    break
                except requests.exceptions.RequestException as e:
                    logging.warning(e)
                    logging.warning("Request Error.")
            else:
                logging.error(f"Cannot download {pathfilename}.")
                raise IOError(f"Cannot download {pathfilename}.")

            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024
            with open(pathfilename, 'wb') as f:
                for chunk in tqdm(
                        r.iter_content(chunk_size=1024),
                        total=math.ceil(total_size // block_size),
                        unit='KB', unit_scale=True,
                        desc=f"Downloading {pathfilename}..."):
                    if chunk:
                        f.write(chunk)
        else:
            logging.error(f"Retry too much times.")
            raise IOError(f"Retry too much times.")
        return pathfilename

    @staticmethod
    def _checksha256(filename, sha256_check):
        if not os.path.isfile(filename):
            return
        h = hashlib.sha256()
        b = bytearray(128*1024)
        mv = memoryview(b)
        with open(filename, 'rb', buffering=0) as f:
            for n in iter(lambda: f.readinto(mv), 0):
                h.update(mv[:n])
        sha256 = h.hexdigest()
        logging.info(f"SHA256 of {filename}: {sha256}")
        if sha256 == sha256_check:
            return True
        logging.warning("SHA256 is not correct.")
        return False
