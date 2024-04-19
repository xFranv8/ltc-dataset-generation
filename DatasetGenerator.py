import carla
import cv2
import numpy as np

import os


class DatasetGenerator:
    def __init__(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.__output_dir = output_dir

    def generate_dataset(self) -> None:
        while True:
            continue

