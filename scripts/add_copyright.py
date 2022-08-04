#!/usr/bin/env python

import argparse
import tempfile
import shutil
from pathlib import Path


def copyright_header(root):
    """ Form the copyright header using c++ style comments. """

    with open(root / 'COPYRIGHT') as f:
        lines = f.readlines()

    return ''.join('\\\\ ' + line for line in lines) + '\n\n'


def file_contains_copyright(file_handle):
    """ Return true if the file already contains a Copyright notice. """

    found = False
    for _, line in zip(range(10), file_handle):
        if line.find('Copyright') != -1:
            found = True

    file_handle.seek(0)
    return found


def add_copyright(header, in_path):
    """ Add copyright notice to file at path. """

    with open(in_path) as f:
        if file_contains_copyright(f):
            print("{} already contains copyright notice".format(in_path))
            return

        # create temporary file holding new text
        out_desc, out_name = tempfile.mkstemp()
        with open(out_desc, 'w') as out:
            out.write(header)
            for line in f:
                out.write(line)

    shutil.copy(out_name, in_path)


parser = argparse.ArgumentParser()
parser.add_argument("--root",
                    type=str,
                    default=".",
                    help="project root")
args = parser.parse_args()

root = Path(args.root)
src = root / 'src'
header = copyright_header(root)

for f in src.rglob('*pp'):
    add_copyright(header, f)
