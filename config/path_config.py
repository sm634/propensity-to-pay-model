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


# Credentials to use GCP
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bold-mantis-312313-85b80c88ad4e.json"