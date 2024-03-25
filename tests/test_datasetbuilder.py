"""Test."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import pytest
import requests
from tqdm.auto import tqdm

import mddatasetbuilder
import mddatasetbuilder.deepmd
import mddatasetbuilder.qmcalc
from mddatasetbuilder._logger import logger

this_directory = os.getcwd()
with open(Path(__file__).parent / "test.json") as f:
    test_params = json.load(f)


class TestMDDatasetBuilder:
    """Test MDDatasetBuilder."""

    @pytest.fixture(params=test_params)
    def datasetbuilder(self, request):
        """DatasetBuilder fixture."""
        folder = tempfile.mkdtemp(prefix="testfiles-", dir=this_directory)
        logger.info(f"Folder: {folder}:")
        os.chdir(folder)
        testparms = request.param
        # download bonds.reaxc and dump.reaxc
        for fileparms in (
            (testparms["bondfile"], testparms["dumpfile"])
            if "bondfile" in testparms
            else (testparms["dumpfile"],)
        ):
            self._download_file(
                fileparms["url"], fileparms["filename"], fileparms["sha256"]
            )

        return mddatasetbuilder.DatasetBuilder(
            bondfilename=testparms["bondfile"]["filename"]
            if "bondfile" in testparms
            else None,
            dumpfilename=testparms["dumpfile"]["filename"],
            atomname=testparms["atomname"],
            n_clusters=testparms["size"],
            dataset_name=testparms["dataset_name"],
            stepinterval=testparms["stepinterval"]
            if "stepinterval" in testparms
            else 1,
        )

    def test_datasetbuilder(self, datasetbuilder):
        """Test DatasetBuilder."""
        datasetbuilder.builddataset()
        assert os.path.exists(datasetbuilder.gjfdir)
        for ii in os.listdir(datasetbuilder.gjfdir):
            mddatasetbuilder.qmcalc.qmcalc(os.path.join(datasetbuilder.gjfdir, ii))
        mddatasetbuilder.deepmd.PrepareDeePMD(datasetbuilder.gjfdir).preparedeepmd()

    def _download_file(self, urls, pathfilename, sha256):
        times = 0
        # download if not exists
        while times < 3:
            if os.path.isfile(pathfilename) and self._checksha256(pathfilename, sha256):
                break
            Path(pathfilename).resolve().parent.mkdir(parents=True, exist_ok=True)

            # from https://stackoverflow.com/questions/16694907
            if not isinstance(urls, list):
                urls = [urls]
            for url in urls:
                try:
                    logger.info(f"Try to download {pathfilename} from {url}")
                    r = requests.get(url, stream=True)
                    break
                except requests.exceptions.RequestException as e:
                    logger.warning(e)
                    logger.warning("Request Error.")
            else:
                logger.error(f"Cannot download {pathfilename}.")
                raise OSError(f"Cannot download {pathfilename}.")

            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            with open(pathfilename, "wb") as f:
                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=math.ceil(total_size // block_size),
                    unit="KB",
                    unit_scale=True,
                    desc=f"Downloading {pathfilename}...",
                    disable=None,
                ):
                    if chunk:
                        f.write(chunk)
        else:
            logger.error("Retry too much times.")
            raise OSError("Retry too much times.")
        return pathfilename

    @staticmethod
    def _checksha256(filename, sha256_check):
        if not os.path.isfile(filename):
            return
        h = hashlib.sha256()
        b = bytearray(128 * 1024)
        mv = memoryview(b)
        with open(filename, "rb", buffering=0) as f:
            for n in iter(lambda: f.readinto(mv), 0):
                h.update(mv[:n])
        sha256 = h.hexdigest()
        logger.info(f"SHA256 of {filename}: {sha256}")
        if sha256 == sha256_check:
            return True
        logger.warning("SHA256 is not correct.")
        return False
