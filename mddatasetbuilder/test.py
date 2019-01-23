'''Test 
python setup.py test
'''


import os
import math
import unittest

from mddatasetbuilder import DatasetBuilder
import requests
from tqdm import tqdm


def download_file(url, local_filename):
    # from https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    with open(local_filename, 'wb') as f:
        for chunk in tqdm(r.iter_content(chunk_size=1024), total=math.ceil(total_size//block_size), unit='KB', unit_scale=True, desc=f"Downloading {local_filename}..."):
            if chunk:
                f.write(chunk)
    return local_filename


class TestReacNetGen(unittest.TestCase):
    def test_reacnetgen(self):
        # download bonds.reaxc and dump.reaxc
        bondfile_url = "https://drive.google.com/uc?authuser=0&id=1CJ22BZTh2Bg3MynHyk_CVZl0rcpSQzRn&export=download"
        dumpfile_url = "https://drive.google.com/uc?authuser=0&id=1-MZZEpTj71JJn4JfKPh5yb_lD2V7NS-Y&export=download"
        folder = "test"
        bondfile = "bonds.reaxc"
        dumpfile = "dump.reaxc"
        bondfilename = os.path.join(folder, bondfile)
        dumpfilename = os.path.join(folder, dumpfile)

        if not os.path.exists(folder):
            os.makedirs(folder)
        print(f"Downloading {bondfile} ...")
        download_file(bondfile_url, bondfilename)
        print(f"Downloading {dumpfile} ...")
        download_file(dumpfile_url, dumpfilename)

        d=DatasetBuilder(bondfilename=bondfilename,dumpfilename=dumpfilename,atomname=["H","O"],dataset_name="h2")
        d.builddataset()

        self.assertTrue(os.path.exists(d.gjfdir))


if __name__ == '__main__':
    unittest.main()
