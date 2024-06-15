import os
import random
import time

import carla
import numpy as np

from CarlaBridge import CarlaBridge
from DatasetGenerator import DatasetGenerator


def main():
    config: dict = {
        "im_width": 256,
        "im_height": 144
    }

    index: int = len(os.listdir("../Dataset/256x144-Dataset"))
    print(index)
    time.sleep(10)

    output_dirs: list[str] = [f"256x144-Dataset/{i}" for i in range(index, 1001)]
    maps: list[str] = ["Town01", "Town02", "Town03", "Town04", "Town05"]

    config["weather"] = carla.WeatherParameters(cloudiness=np.random.normal(25, 50),
                                                precipitation=np.random.normal(0, 20),
                                                sun_altitude_angle=np.random.normal(60, 25))

    config["town"] = random.choice(maps)

    for dir in output_dirs:
        print(f"Generating {dir} sequence")

        config["weather"] = carla.WeatherParameters(cloudiness=np.random.normal(25, 50),
                                                    precipitation=np.random.normal(0, 20),
                                                    sun_altitude_angle=np.random.normal(60, 25))

        config["town"] = random.choice(maps)

        carla_bridge: CarlaBridge = CarlaBridge(config)
        generator: DatasetGenerator = DatasetGenerator(dir, carla_bridge)

        generator.generate_dataset()
        generator.destroy_actors()
        time.sleep(10)


if __name__ == '__main__':
    main()
