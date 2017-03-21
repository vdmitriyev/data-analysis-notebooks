#!/usr/bin/env python3

import sys
import os.path
import zipfile
from urllib.request import urlretrieve

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

def start_download():

    base_url = 'http://www.nrel.gov/midc/oahu_archive/rawdata/Oahu_GHI/'

    zip_files = ['201003', '201004', '201005', '201006', '201007', '201008', '201009', '201010', '201011', '201012',
                 '201101', '201102', '201103', '201104', '201105', '201106', '201107', '201108', '201109', '201110']

    for zip_f in zip_files:
        target_zip = zip_f + '.zip'
        target_url = base_url + target_zip

        if not os.path.exists(target_zip):
            print('Downloading ... File: {0}'.format(target_zip))
            urlretrieve(target_url, target_zip, reporthook)
            print('Downloaded successfully . File: {0}'.format(target_zip))
        else:
            print('Skipping ... File already existing: {0}'.format(target_zip))

        print('Unzipping ... File: {0}'.format(target_zip))
        zfile = zipfile.ZipFile(target_zip)
        zfile.extractall(zip_f)

if __name__ == "__main__":
    start_download()
