from os import getenv
import logging
from typing import NoReturn, Union
import sys
import rtoml
import os
import sklearn
import numpy as np
import math

from configs.config import (XML_FILE_PATH, TXT_OUTPUT_PATH)


class GeneralFunctions(object):
    def __init__(self) -> None:
        pass
    
    def local_logger(self, file_path: str='logs/debug.log'):
        """ Set up a local logger

        Args:
            file_path (str, optional): _description_. Defaults to 'logs/debug.log'.
        """
        local_logging = getenv(key="LOCAL_LOGGING", default=False)
        if local_logging:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(file_path),
                    logging.StreamHandler()
                ]
            )
        return
        
    def load_toml(self, toml_file_path: str) -> Union[dict, NoReturn]:
        """ Load a toml file and return it as a dictionary

        Args:
            toml_file_path (str): _description_

        Returns:
            Union[dict, NoReturn]: _description_
        """
        try:
            f = open(toml_file_path, 'r')
            toml_loaded_dict = rtoml.load(f.read())
            return toml_loaded_dict
        except Exception as e:
            logging.error(e)
            return sys.exit(1)
        
    def normalize(self, x: np.arrary, axis: str=0) -> np.array:
        """ Normalise the spectral centroid for visualisation

        Args:
            x (np.arrary): _description_
            axis (str, optional): _description_. Defaults to 0.

        Returns:
            np.array: _description_
        """
        return sklearn.preprocessing.minmax_scale(x, axis=axis)
    
    def get_filepaths(self, directory) -> list:
        """ Get all file paths in a directory

        Args:
            directory (_type_): _description_

        Returns:
            list: _description_
        """
        file_paths = []  # List which will store all of the full filepaths.
        # Walk the tree.
        for root, directories, files in os.walk(directory):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

        return file_paths
    
    def get_lip_height(self, lip: list) -> float:
        """ Get the height of the lip

        Args:
            lip (list): _description_

        Returns:
            float: _description_
        """
        sum=0
        for i in [2,3,4]:
            # distance between two near points up and down
            distance = math.sqrt( (lip[i][0] - lip[12-i][0])**2 +
                                (lip[i][1] - lip[12-i][1])**2   )
            sum += distance
        return sum / 3
    
    def get_mouth_height(self, top_lip: list, bottom_lip: list) -> float:
        """ Get the height of the mouth

        Args:
            top_lip (list): _description_
            bottom_lip (list): _description_

        Returns:
            float: _description_
        """
        sum=0
        for i in [8,9,10]:
            # distance between two near points up and down
            distance = math.sqrt( (top_lip[i][0] - bottom_lip[18-i][0])**2 +
                                (top_lip[i][1] - bottom_lip[18-i][1])**2   )
            sum += distance
        return sum / 3