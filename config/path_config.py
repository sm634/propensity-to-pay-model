import os
from sys import path


def extend_dir_path():
    cwd = os.getcwd()
    preprocessing_path = cwd + '\\preprocessing_layer\\'
    path.extend(
        [
            preprocessing_path
        ]
    )
