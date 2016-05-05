#/usr/bin/python
import os

__doc__="""a configuration file that stores paths and constants used by other notebooks"""


path_to_data = "/srv/hd7/jheuristic/ksfinder_data"
from_data = lambda path: os.path.join(path_to_data, path)


